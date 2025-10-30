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


    // ========== WHATSAPP FLOW EVENTS ==========

    /** Nhận được lệnh WhatsApp từ Speech Recognition */
    data class WACommandReceived(val rawCommand: String) : VoiceEvent()

    /** Lệnh WhatsApp được phân tích thành công */
    data class WACommandParsed(
        val receiver: String,
        val message: String
    ) : VoiceEvent()

    /** Lệnh WhatsApp không thể phân tích (parsing failed) */
    data class WACommandParseFailed(val reason: String) : VoiceEvent()

    /** Người dùng xác nhận lệnh WhatsApp (có/không) */
    data class WAUserConfirmed(val confirmed: Boolean) : VoiceEvent()

    /** Tìm thấy liên hệ chính xác 100% cho WhatsApp */
    data class WAExactContactFound(
        val contactName: String,
        val phoneNumber: String
    ) : VoiceEvent()

    /** Tìm thấy nhiều liên hệ tương tự cho WhatsApp */
    data class WASimilarContactsFound(
        val originalName: String,
        val similarContacts: List<String>
    ) : VoiceEvent()

    /** Không tìm thấy liên hệ nào cho WhatsApp */
    data class WANoContactFound(val searchName: String) : VoiceEvent()

    /** Người dùng cung cấp tên liên hệ mới cho WhatsApp */
    data class WANewContactNameProvided(val newName: String) : VoiceEvent()

    /** Người dùng từ chối thử lại WhatsApp (phủ định) */
    object WAUserDeclinedRetry : VoiceEvent()

    /** WhatsApp được gửi thành công */
    object WASentSuccessfully : VoiceEvent()

    /** Gửi WhatsApp thất bại */
    data class WASendFailed(val error: String) : VoiceEvent()


    // ========== SPEECH RECOGNITION EVENTS ==========

    /** Speech Recognition không nhận dạng được giọng nói */
    object SpeechRecognitionFailed : VoiceEvent()

    /** Speech Recognition nhận được kết quả nhưng không rõ ràng */
    data class UnclearSpeechResult(val possibleResults: List<String>) : VoiceEvent()


    // ========== PHONE CALL EVENTS ==========

    /** Nhận được lệnh gọi điện */
    data class CallCommandReceived(val rawCommand: String) : VoiceEvent()

    /** Lệnh gọi điện được phân tích thành công */
    data class CallCommandParsed(val contactName: String) : VoiceEvent()

    /** Lệnh gọi điện không thể phân tích (parsing failed) */
    data class CallCommandParseFailed(val reason: String) : VoiceEvent()

    /** Cuộc gọi được thực hiện thành công */
    object CallMadeSuccessfully : VoiceEvent()

    /** Cuộc gọi thất bại */
    data class CallFailed(val error: String) : VoiceEvent()


    // ========== CHROME SEARCH EVENTS ==========

    /** Nhận được lệnh tìm kiếm Chrome */
    data class ChromeCommandReceived(val rawCommand: String) : VoiceEvent()

    /** Lệnh tìm kiếm Chrome được phân tích thành công */
    data class ChromeCommandParsed(val query: String) : VoiceEvent()

    /** Lệnh tìm kiếm Chrome không thể phân tích (parsing failed) */
    data class ChromeCommandParseFailed(val reason: String) : VoiceEvent()

    /** Tìm kiếm Chrome thành công */
    object ChromeSearchSuccessfully : VoiceEvent()

    /** Tìm kiếm Chrome thất bại */
    data class ChromeSearchFailed(val error: String) : VoiceEvent()


    // ========== YOUTUBE SEARCH EVENTS ==========

    /** Nhận được lệnh tìm kiếm YouTube */
    data class YouTubeCommandReceived(val rawCommand: String) : VoiceEvent()

    /** Lệnh tìm kiếm YouTube được phân tích thành công */
    data class YouTubeCommandParsed(val query: String) : VoiceEvent()

    /** Lệnh tìm kiếm YouTube không thể phân tích (parsing failed) */
    data class YouTubeCommandParseFailed(val reason: String) : VoiceEvent()

    /** Tìm kiếm YouTube thành công */
    object YouTubeSearchSuccessfully : VoiceEvent()

    /** Tìm kiếm YouTube thất bại */
    data class YouTubeSearchFailed(val error: String) : VoiceEvent()


    // ========== ALARM EVENTS ==========

    /** Bắt đầu lệnh tạo báo thức */
    object StartAlarmCommand : VoiceEvent()
    object StartChromeCommand : VoiceEvent()
    object StartYouTubeCommand : VoiceEvent()
    object StartSMSCommand : VoiceEvent()
    object SMSConfirmed : VoiceEvent()
    object SMSCancelled : VoiceEvent()
    object StartWACommand : VoiceEvent()
    object WAConfirmed : VoiceEvent()
    object WACancelled : VoiceEvent()
    object StartPhoneCommand : VoiceEvent()
    object PhoneConfirmed : VoiceEvent()
    object PhoneCancelled : VoiceEvent()

    /** Nhận được lệnh tạo báo thức */
    data class AlarmCommandReceived(val rawCommand: String) : VoiceEvent()

    /** Lệnh tạo báo thức được phân tích thành công */
    data class AlarmCommandParsed(val hour: Int, val minute: Int, val message: String) : VoiceEvent()

    /** Lệnh tạo báo thức không thể phân tích (parsing failed) */
    data class AlarmCommandParseFailed(val reason: String) : VoiceEvent()

    /** Tạo báo thức thành công */
    object AlarmCreatedSuccessfully : VoiceEvent()

    /** Tạo báo thức thất bại */
    data class AlarmCreationFailed(val error: String) : VoiceEvent()


    // ========== CALENDAR EVENTS ==========

    /** Nhận được lệnh tạo sự kiện lịch */
    data class CalendarCommandReceived(val rawCommand: String) : VoiceEvent()

    /** Lệnh tạo sự kiện lịch được phân tích thành công */
    data class CalendarCommandParsed(val title: String, val location: String, val begin: Long, val end: Long) : VoiceEvent()

    /** Lệnh tạo sự kiện lịch không thể phân tích (parsing failed) */
    data class CalendarCommandParseFailed(val reason: String) : VoiceEvent()

    /** Tạo sự kiện lịch thành công */
    object CalendarEventCreatedSuccessfully : VoiceEvent()

    /** Tạo sự kiện lịch thất bại */
    data class CalendarEventCreationFailed(val error: String) : VoiceEvent()

    // ========== CAMERA EVENTS ==========
    object StartCameraCapture : VoiceEvent()
    data class CameraCommandReceived(val rawCommand: String) : VoiceEvent()
    data class CameraCommandParsed(val action: String) : VoiceEvent()
    data class CameraCommandParseFailed(val reason: String) : VoiceEvent()
    object CameraCapturedSuccessfully : VoiceEvent()
    data class CameraCaptureFailed(val error: String) : VoiceEvent()

    // ========== DEVICE EVENTS ==========
    object StartFlashCommand : VoiceEvent()
    object StartWifiCommand : VoiceEvent()
    object StartVolumeCommand : VoiceEvent()
    
    object FlashToggledSuccessfully : VoiceEvent()
    data class FlashToggleFailed(val error: String) : VoiceEvent()
    
    object WifiToggledSuccessfully : VoiceEvent()
    data class WifiToggleFailed(val error: String) : VoiceEvent()
    
    object VolumeAdjustedSuccessfully : VoiceEvent()
    data class VolumeAdjustmentFailed(val error: String) : VoiceEvent()
    
    // ========== UTILITY EVENTS ==========
    object Reset : VoiceEvent()


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
            is WACommandReceived -> "WACommandReceived(${rawCommand.take(50)}...)"
            is WACommandParsed -> "WACommandParsed($receiver, ${message.take(20)}...)"
            is WACommandParseFailed -> "WACommandParseFailed($reason)"
            is WAUserConfirmed -> "WAUserConfirmed($confirmed)"
            is WAExactContactFound -> "WAExactContactFound($contactName, $phoneNumber)"
            is WASimilarContactsFound -> "WASimilarContactsFound(${similarContacts.size} contacts)"
            is WANoContactFound -> "WANoContactFound($searchName)"
            is WANewContactNameProvided -> "WANewContactNameProvided($newName)"
            is WAUserDeclinedRetry -> "WAUserDeclinedRetry"
            is WASentSuccessfully -> "WASentSuccessfully"
            is WASendFailed -> "WASendFailed($error)"
            is SpeechRecognitionFailed -> "SpeechRecognitionFailed"
            is UnclearSpeechResult -> "UnclearSpeechResult(${possibleResults.size} results)"
            is CallCommandReceived -> "CallCommandReceived(${rawCommand.take(50)}...)"
            is CallCommandParsed -> "CallCommandParsed($contactName)"
            is CallCommandParseFailed -> "CallCommandParseFailed($reason)"
            is CallMadeSuccessfully -> "CallMadeSuccessfully"
            is CallFailed -> "CallFailed($error)"
            is ChromeCommandReceived -> "ChromeCommandReceived(${rawCommand.take(50)}...)"
            is ChromeCommandParsed -> "ChromeCommandParsed($query)"
            is ChromeCommandParseFailed -> "ChromeCommandParseFailed($reason)"
            is ChromeSearchSuccessfully -> "ChromeSearchSuccessfully"
            is ChromeSearchFailed -> "ChromeSearchFailed($error)"
            is YouTubeCommandReceived -> "YouTubeCommandReceived(${rawCommand.take(50)}...)"
            is YouTubeCommandParsed -> "YouTubeCommandParsed($query)"
            is YouTubeCommandParseFailed -> "YouTubeCommandParseFailed($reason)"
            is YouTubeSearchSuccessfully -> "YouTubeSearchSuccessfully"
            is YouTubeSearchFailed -> "YouTubeSearchFailed($error)"
            is StartAlarmCommand -> "StartAlarmCommand"
            is StartChromeCommand -> "StartChromeCommand"
            is StartYouTubeCommand -> "StartYouTubeCommand"
            is StartSMSCommand -> "StartSMSCommand"
        is SMSConfirmed -> "SMSConfirmed"
        is SMSCancelled -> "SMSCancelled"
        is StartWACommand -> "StartWACommand"
        is WAConfirmed -> "WAConfirmed"
        is WACancelled -> "WACancelled"
        is StartPhoneCommand -> "StartPhoneCommand"
        is PhoneConfirmed -> "PhoneConfirmed"
        is PhoneCancelled -> "PhoneCancelled"
            is AlarmCommandReceived -> "AlarmCommandReceived(${rawCommand.take(50)}...)"
            is AlarmCommandParsed -> "AlarmCommandParsed($hour:$minute - $message)"
            is AlarmCommandParseFailed -> "AlarmCommandParseFailed($reason)"
            is AlarmCreatedSuccessfully -> "AlarmCreatedSuccessfully"
            is AlarmCreationFailed -> "AlarmCreationFailed($error)"
            is CalendarCommandReceived -> "CalendarCommandReceived(${rawCommand.take(50)}...)"
            is CalendarCommandParsed -> "CalendarCommandParsed($title)"
            is CalendarCommandParseFailed -> "CalendarCommandParseFailed($reason)"
            is CalendarEventCreatedSuccessfully -> "CalendarEventCreatedSuccessfully"
            is CalendarEventCreationFailed -> "CalendarEventCreationFailed($error)"
            
            // Device events
            is StartFlashCommand -> "StartFlashCommand"
            is StartWifiCommand -> "StartWifiCommand"
            is StartVolumeCommand -> "StartVolumeCommand"
            is FlashToggledSuccessfully -> "FlashToggledSuccessfully"
            is FlashToggleFailed -> "FlashToggleFailed($error)"
            is WifiToggledSuccessfully -> "WifiToggledSuccessfully"
            is WifiToggleFailed -> "WifiToggleFailed($error)"
            is VolumeAdjustedSuccessfully -> "VolumeAdjustedSuccessfully"
            is VolumeAdjustmentFailed -> "VolumeAdjustmentFailed($error)"
            
            // Device Control Events
            is StartCameraCapture -> "StartCameraCapture"
            is CameraCommandReceived -> "CameraCommandReceived($rawCommand)"
            is CameraCommandParsed -> "CameraCommandParsed($action)"
            is CameraCommandParseFailed -> "CameraCommandParseFailed($reason)"
            is CameraCapturedSuccessfully -> "CameraCapturedSuccessfully"
            is CameraCaptureFailed -> "CameraCaptureFailed($error)"
            is Reset -> "Reset"
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
                this is WACommandParseFailed ||
                this is WANoContactFound ||
                this is WASendFailed ||
                this is SpeechRecognitionFailed ||
                this is CallCommandParseFailed ||
                this is CallFailed ||
                this is ChromeCommandParseFailed ||
                this is ChromeSearchFailed ||
                this is YouTubeCommandParseFailed ||
                this is YouTubeSearchFailed ||
                this is AlarmCommandParseFailed ||
                this is AlarmCreationFailed ||
                this is CalendarCommandParseFailed ||
                this is CalendarEventCreationFailed ||
                // Device Control Error Events
                this is CameraCommandParseFailed ||
                this is CameraCaptureFailed ||
                this is WifiToggleFailed ||
                this is VolumeAdjustmentFailed ||
                this is FlashToggleFailed
    }
}