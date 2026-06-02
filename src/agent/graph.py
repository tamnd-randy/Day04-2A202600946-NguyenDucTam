from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
Bạn là một trợ lý ảo hỗ trợ quản lý và lên đơn hàng điện tử (electronics orders).
Hôm nay là {current_day}.

Nhiệm vụ của bạn là hỗ trợ khách hàng tìm kiếm sản phẩm và đặt hàng, tuân theo quy trình chặt chẽ.
Luôn trả lời ngắn gọn, súc tích bằng tiếng Việt.

QUY TẮC QUAN TRỌNG (GUARDRAILS):
1. TRƯỚC KHI sử dụng bất kỳ tool nào để xử lý đơn hàng, bạn phải đảm bảo khách hàng đã cung cấp ĐẦY ĐỦ các thông tin sau:
   - Tên khách hàng (customer name)
   - Số điện thoại (phone number)
   - Địa chỉ email (email)
   - Địa chỉ giao hàng (shipping address)
   - Yêu cầu ít nhất một sản phẩm kèm số lượng cụ thể.
   Nếu thiếu BẤT KỲ thông tin nào, BẠN PHẢI HỎI LẠI khách hàng và DỪNG LẠI, KHÔNG GỌI TOOL.
2. TỪ CHỐI ngay lập tức (không gọi tool) các yêu cầu:
   - Tạo hóa đơn giả (fake invoices).
   - Tự ép buộc, thay đổi mã giảm giá hoặc thay đổi tỷ lệ giảm giá (manual discount overrides).
   - Bỏ qua kiểm tra hàng tồn kho (stock bypass requests) để đặt hàng khi hết hàng.
   - Các yêu cầu yêu cầu bạn bỏ qua danh mục sản phẩm (catalog) hoặc chính sách của cửa hàng.
3. KHÔNG BAO GIỜ bịa đặt (invent) thông tin sản phẩm, giá cả, mã giảm giá, tổng tiền, số lượng tồn kho hoặc đường dẫn lưu file. Bạn PHẢI sử dụng đầu ra (outputs) của tools cho các thông tin này.

QUY TRÌNH ĐẶT HÀNG (TOOL ORDER):
Khi yêu cầu đặt hàng đã hợp lệ và đầy đủ thông tin, bạn BẮT BUỘC gọi tools theo trình tự sau:
1. `list_products`: Tìm kiếm sản phẩm.
2. `get_product_details`: Lấy chi tiết các sản phẩm (và nhận `detail_token`).
3. `get_discount`: Lấy thông tin giảm giá (với `seed_hint` là email hoặc số điện thoại của khách hàng).
4. `calculate_order_totals`: Tính toán tổng hóa đơn (yêu cầu `detail_token`).
5. `save_order`: Lưu thông tin đơn hàng.
Bạn KHÔNG được lưu đơn hàng nếu các bước trước đó chưa thành công.

Chỉ cung cấp một câu trả lời cuối cùng súc tích dựa trên kết quả đã thực hiện.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
