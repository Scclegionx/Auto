package com.auto_fe.auto_fe.base

/**
 * Exception đặc biệt để signal rằng cần xác nhận từ người dùng trước khi thực thi
 * @param originalInput Câu nói gốc của người dùng
 * @param confirmationQuestion Câu hỏi xác nhận
 * @param onConfirmed Callback để thực thi khi được xác nhận
 * @param isMultipleContacts Nếu true, nghĩa là có nhiều liên hệ và cần người dùng nói tên
 * @param actionType Loại hành động: "sms" hoặc "phone"
 * @param actionData Dữ liệu bổ sung cho hành động:
 *   - Với SMS: nội dung tin nhắn
 *   - Với Phone: platform (phone, zalo, etc.)
 */
class ConfirmationRequirement(
    val originalInput: String,
    val confirmationQuestion: String,
    val onConfirmed: suspend () -> String,
    val isMultipleContacts: Boolean = false,
    val actionType: String = "",
    val actionData: String = ""
) : Exception("Confirmation required")

