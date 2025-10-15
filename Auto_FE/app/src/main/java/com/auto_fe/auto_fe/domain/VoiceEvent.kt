package com.auto_fe.auto_fe.domain

/**
 * Định nghĩa tất cả các sự kiện (events) có thể xảy ra trong Voice Assistant
 * Events trigger state transitions
 */
sealed class VoiceEvent {

    // ========== COMMON EVENTS ==========

    /** Người dùng bấm nút bắt đầu ghi âm */
    object StartRecording : VoiceEvent()

    /** Timeout - không nhận được phản hồi từ người dùng */
    object Timeout : VoiceEvent()

    /** Người dùng hủy lệnh */
    object UserCancelled : VoiceEvent()

    /** Lỗi quyền truy cập (permission denied) */
    data class PermissionError(val permission: String) : VoiceEvent()


    // ========== SMS FLOW EVENTS ==========

    /** Nhận được lệnh SMS từ Speech Recognition */
    data class SMSCommandReceived(val rawCommand: String) : VoiceEvent()

    /** Lệnh SMS được phân tích thành công */
    data class SMSCommandParsed(
        val receiver: String,
        val message: String
    ) : VoiceEvent()

    /** Lệnh SMS không thể phân tích (parsing failed) */
    data class SMSCommandParseFailed(val reason: String) : VoiceEvent()

    /** Người dùng xác nhận lệnh (có/không) */
    data class UserConfirmed(val confirmed: Boolean) : VoiceEvent()

    /** Tìm thấy liên hệ chính xác 100% */
    data class ExactContactFound(
        val contactName: String,
        val phoneNumber: String
    ) : VoiceEvent()

    /** Tìm thấy nhiều liên hệ tương tự */
    data class SimilarContactsFound(
        val originalName: String,
        val similarContacts: List<String>
    ) : VoiceEvent()

    /** Không tìm thấy liên hệ nào */
    data class NoContactFound(val searchName: String) : VoiceEvent()

    /** Người dùng cung cấp tên liên hệ mới */
    data class NewContactNameProvided(val newName: String) : VoiceEvent()

    /** Người dùng từ chối thử lại (phủ định) */
    object UserDeclinedRetry : VoiceEvent()

    /** SMS được gửi thành công */
    object SMSSentSuccessfully : VoiceEvent()

    /** Gửi SMS thất bại */
    data class SMSSendFailed(val error: String) : VoiceEvent()


    // ========== SPEECH RECOGNITION EVENTS ==========

    /** Speech Recognition không nhận dạng được giọng nói */
    object SpeechRecognitionFailed : VoiceEvent()

    /** Speech Recognition nhận được kết quả nhưng không rõ ràng */
    data class UnclearSpeechResult(val possibleResults: List<String>) : VoiceEvent()


    // ========== FUTURE: PHONE CALL EVENTS (để sau) ==========

    /** Nhận được lệnh gọi điện */
    data class CallCommandReceived(val contactName: String) : VoiceEvent()

    /** Cuộc gọi được thực hiện thành công */
    object CallMadeSuccessfully : VoiceEvent()

    /** Cuộc gọi thất bại */
    data class CallFailed(val error: String) : VoiceEvent()


    // ========== UTILITY METHODS ==========

    /**
     * Lấy tên event để logging
     */
    fun getName(): String {
        return when (this) {
            is StartRecording -> "StartRecording"
            is Timeout -> "Timeout"
            is UserCancelled -> "UserCancelled"
            is PermissionError -> "PermissionError($permission)"
            is SMSCommandReceived -> "SMSCommandReceived(${rawCommand.take(50)}...)"
            is SMSCommandParsed -> "SMSCommandParsed($receiver, ${message.take(20)}...)"
            is SMSCommandParseFailed -> "SMSCommandParseFailed($reason)"
            is UserConfirmed -> "UserConfirmed($confirmed)"
            is ExactContactFound -> "ExactContactFound($contactName, $phoneNumber)"
            is SimilarContactsFound -> "SimilarContactsFound(${similarContacts.size} contacts)"
            is NoContactFound -> "NoContactFound($searchName)"
            is NewContactNameProvided -> "NewContactNameProvided($newName)"
            is UserDeclinedRetry -> "UserDeclinedRetry"
            is SMSSentSuccessfully -> "SMSSentSuccessfully"
            is SMSSendFailed -> "SMSSendFailed($error)"
            is SpeechRecognitionFailed -> "SpeechRecognitionFailed"
            is UnclearSpeechResult -> "UnclearSpeechResult(${possibleResults.size} results)"
            is CallCommandReceived -> "CallCommandReceived($contactName)"
            is CallMadeSuccessfully -> "CallMadeSuccessfully"
            is CallFailed -> "CallFailed($error)"
        }
    }

    /**
     * Kiểm tra xem event có phải là error event không
     */
    fun isError(): Boolean {
        return this is PermissionError ||
                this is SMSCommandParseFailed ||
                this is NoContactFound ||
                this is SMSSendFailed ||
                this is SpeechRecognitionFailed ||
                this is CallFailed
    }
}