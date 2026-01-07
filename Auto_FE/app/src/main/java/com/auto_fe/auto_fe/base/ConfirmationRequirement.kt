package com.auto_fe.auto_fe.base

/**
 * Exception đặc biệt để signal rằng cần xác nhận từ người dùng trước khi thực thi
 * @param originalInput Câu nói gốc của người dùng
 * @param confirmationQuestion Câu hỏi xác nhận
 * @param onConfirmed Callback để thực thi khi được xác nhận
 */
class ConfirmationRequirement(
    val originalInput: String,
    val confirmationQuestion: String,
    val onConfirmed: suspend () -> String
) : Exception("Confirmation required")

