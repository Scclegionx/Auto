package com.auto_fe.auto_fe.domain

/**
 * Định nghĩa tất cả các trạng thái của Voice Assistant
 * Mỗi state đại diện cho một bước trong luồng xử lý
 */
sealed class VoiceState {

    // ========== COMMON STATES ==========

    /** Trạng thái ban đầu - chờ người dùng tương tác */
    object Idle : VoiceState()

    /** Trạng thái kết thúc thành công */
    object Success : VoiceState()

    /** Trạng thái lỗi hoặc hủy bỏ */
    data class Error(val errorMessage: String = "") : VoiceState()


    // ========== SMS FLOW STATES ==========

    /** Đang lắng nghe lệnh gửi SMS từ người dùng */
    object ListeningForSMSCommand : VoiceState()

    /** Đang phân tích lệnh SMS (parsing command) */
    object ParsingSMSCommand : VoiceState()

    /** Đang xác nhận lệnh với người dùng */
    data class ConfirmingSMSCommand(
        val receiver: String,
        val message: String
    ) : VoiceState()

    /** Đang chờ người dùng xác nhận (có/không) */
    object WaitingForUserConfirmation : VoiceState()

    /** Đang tìm kiếm liên hệ trong danh bạ */
    data class SearchingContact(
        val contactName: String,
        val message: String
    ) : VoiceState()

    /** Tìm thấy nhiều liên hệ tương tự - cần xác nhận */
    data class SuggestingSimilarContacts(
        val originalName: String,
        val similarContacts: List<String>,
        val message: String,
        val retryCount: Int = 0
    ) : VoiceState()

    /** Đang chờ người dùng cung cấp tên mới */
    object WaitingForNewContactName : VoiceState()

    /** Đang gửi tin nhắn */
    data class SendingSMS(
        val phoneNumber: String,
        val message: String,
        val contactName: String
    ) : VoiceState()


    // ========== FUTURE: PHONE CALL STATES (để sau) ==========

    /** Đang lắng nghe lệnh gọi điện */
    object ListeningForCallCommand : VoiceState()

    /** Đang thực hiện cuộc gọi */
    data class MakingCall(val phoneNumber: String) : VoiceState()


    // ========== UTILITY METHODS ==========

    /**
     * Kiểm tra xem state có phải là terminal state không
     * Terminal states là những state kết thúc luồng (Success, Error)
     */
    fun isTerminal(): Boolean {
        return this is Success || this is Error
    }

    /**
     * Kiểm tra xem state có phải đang trong SMS flow không
     */
    fun isSMSFlow(): Boolean {
        return this is ListeningForSMSCommand ||
                this is ParsingSMSCommand ||
                this is ConfirmingSMSCommand ||
                this is WaitingForUserConfirmation ||
                this is SearchingContact ||
                this is SuggestingSimilarContacts ||
                this is WaitingForNewContactName ||
                this is SendingSMS
    }

    /**
     * Lấy tên state để logging
     */
    fun getName(): String {
        return when (this) {
            is Idle -> "Idle"
            is Success -> "Success"
            is Error -> "Error: $errorMessage"
            is ListeningForSMSCommand -> "ListeningForSMSCommand"
            is ParsingSMSCommand -> "ParsingSMSCommand"
            is ConfirmingSMSCommand -> "ConfirmingSMSCommand($receiver, $message)"
            is WaitingForUserConfirmation -> "WaitingForUserConfirmation"
            is SearchingContact -> "SearchingContact($contactName)"
            is SuggestingSimilarContacts -> "SuggestingSimilarContacts(${similarContacts.size} contacts, retry=$retryCount)"
            is WaitingForNewContactName -> "WaitingForNewContactName"
            is SendingSMS -> "SendingSMS($contactName)"
            is ListeningForCallCommand -> "ListeningForCallCommand"
            is MakingCall -> "MakingCall($phoneNumber)"
        }
    }
}