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

    /** Trạng thái hủy bỏ */
    object Cancel : VoiceState()

    /** Trạng thái lỗi */
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


    // ========== WHATSAPP FLOW STATES ==========

    /** Đang lắng nghe lệnh gửi WhatsApp từ người dùng */
    object ListeningForWACommand : VoiceState()

    /** Đang phân tích lệnh WhatsApp (parsing command) */
    object ParsingWACommand : VoiceState()

    /** Đang xác nhận lệnh với người dùng */
    data class ConfirmingWACommand(
        val receiver: String,
        val message: String
    ) : VoiceState()

    /** Đang chờ người dùng xác nhận WhatsApp (có/không) */
    object WaitingForWAUserConfirmation : VoiceState()

    /** Đang tìm kiếm liên hệ trong danh bạ cho WhatsApp */
    data class SearchingWAContact(
        val contactName: String,
        val message: String
    ) : VoiceState()

    /** Tìm thấy nhiều liên hệ tương tự cho WhatsApp - cần xác nhận */
    data class SuggestingWASimilarContacts(
        val originalName: String,
        val similarContacts: List<String>,
        val message: String,
        val retryCount: Int = 0
    ) : VoiceState()

    /** Đang chờ người dùng cung cấp tên mới cho WhatsApp */
    object WaitingForWANewContactName : VoiceState()

    /** Đang gửi tin nhắn WhatsApp */
    data class SendingWA(
        val phoneNumber: String,
        val message: String,
        val contactName: String
    ) : VoiceState()

    /** Đang thực hiện lệnh WhatsApp */
    object ExecutingWACommand : VoiceState()


    // ========== PHONE CALL STATES ==========

    /** Đang lắng nghe lệnh gọi điện */
    object ListeningForCallCommand : VoiceState()

    /** Đang phân tích lệnh gọi điện (parsing command) */
    object ParsingCallCommand : VoiceState()

    /** Đang thực hiện cuộc gọi */
    data class MakingCall(val phoneNumber: String) : VoiceState()


    // ========== CHROME SEARCH STATES ==========

    /** Đang lắng nghe lệnh tìm kiếm Chrome */
    object ListeningForChromeCommand : VoiceState()

    /** Đang phân tích lệnh tìm kiếm Chrome (parsing command) */
    object ParsingChromeCommand : VoiceState()

    /** Đang thực hiện tìm kiếm Chrome */
    data class SearchingChrome(val query: String) : VoiceState()


    // ========== YOUTUBE SEARCH STATES ==========

    /** Đang lắng nghe lệnh tìm kiếm YouTube */
    object ListeningForYouTubeCommand : VoiceState()

    /** Đang phân tích lệnh tìm kiếm YouTube (parsing command) */
    object ParsingYouTubeCommand : VoiceState()

    /** Đang thực hiện tìm kiếm YouTube */
    data class SearchingYouTube(val query: String) : VoiceState()


    // ========== ALARM STATES ==========

    /** Đang lắng nghe lệnh tạo báo thức */
    object ListeningForAlarmCommand : VoiceState()

    /** Đang phân tích lệnh tạo báo thức (parsing command) */
    object ParsingAlarmCommand : VoiceState()

    /** Đang thực hiện tạo báo thức */
    object ExecutingAlarmCommand : VoiceState()
    object ExecutingChromeCommand : VoiceState()
    object ExecutingYouTubeCommand : VoiceState()
    object ExecutingSMSCommand : VoiceState()
    object ConfirmingPhoneCommand : VoiceState()
    object ExecutingPhoneCommand : VoiceState()

    /** Đang thực hiện tạo báo thức */
    data class CreatingAlarm(val hour: Int, val minute: Int, val message: String) : VoiceState()


    // ========== CALENDAR STATES ==========

    /** Đang lắng nghe lệnh tạo sự kiện lịch */
    object ListeningForCalendarCommand : VoiceState()

    /** Đang phân tích lệnh tạo sự kiện lịch (parsing command) */
    object ParsingCalendarCommand : VoiceState()

    /** Đang thực hiện tạo sự kiện lịch */
    data class CreatingCalendarEvent(val title: String, val location: String, val begin: Long, val end: Long) : VoiceState()

    // ========== CAMERA STATES ==========
    object ListeningForCameraCommand : VoiceState()
    object ParsingCameraCommand : VoiceState()
    object ExecutingCameraCommand : VoiceState()

    // ========== FLASH STATES ==========
    object ListeningForFlashCommand : VoiceState()
    object ParsingFlashCommand : VoiceState()
    object ExecutingFlashCommand : VoiceState()

    // ========== WIFI STATES ==========
    object ListeningForWifiCommand : VoiceState()
    object ParsingWifiCommand : VoiceState()
    object ExecutingWifiCommand : VoiceState()

    // ========== VOLUME STATES ==========
    object ListeningForVolumeCommand : VoiceState()
    object ParsingVolumeCommand : VoiceState()
    object ExecutingVolumeCommand : VoiceState()


    // ========== ADD CONTACT STATES ==========
    /** Bắt đầu luồng thêm liên hệ: hỏi tên */
    object AskingContactName : VoiceState()
    /** Hỏi số điện thoại sau khi có tên */
    data class AskingContactPhone(val contactName: String) : VoiceState()
    /** Đang thực thi tạo liên hệ */
    object ExecutingAddContactCommand : VoiceState()


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
     * Kiểm tra xem state có phải đang trong Phone flow không
     */
    fun isPhoneFlow(): Boolean {
        return this is ListeningForCallCommand ||
                this is ParsingCallCommand ||
                this is MakingCall
    }

    /**
     * Kiểm tra xem state có phải đang trong Chrome flow không
     */
    fun isChromeFlow(): Boolean {
        return this is ListeningForChromeCommand ||
                this is ParsingChromeCommand ||
                this is SearchingChrome
    }

    /**
     * Kiểm tra xem state có phải đang trong YouTube flow không
     */
    fun isYouTubeFlow(): Boolean {
        return this is ListeningForYouTubeCommand ||
                this is ParsingYouTubeCommand ||
                this is SearchingYouTube
    }

    /**
     * Kiểm tra xem state có phải đang trong Alarm flow không
     */
    fun isAlarmFlow(): Boolean {
        return this is ListeningForAlarmCommand ||
                this is ParsingAlarmCommand ||
                this is CreatingAlarm
    }

    /**
     * Kiểm tra xem state có phải đang trong Calendar flow không
     */
    fun isCalendarFlow(): Boolean {
        return this is ListeningForCalendarCommand || 
               this is ParsingCalendarCommand || 
               this is CreatingCalendarEvent
    }

    fun isCameraFlow(): Boolean {
        return this is ListeningForCameraCommand || 
               this is ParsingCameraCommand ||
               this is ExecutingCameraCommand
    }

    fun isWifiFlow(): Boolean {
        return this is ListeningForWifiCommand || 
               this is ParsingWifiCommand
    }

    fun isVolumeFlow(): Boolean {
        return this is ListeningForVolumeCommand || 
               this is ParsingVolumeCommand
    }

    fun isFlashFlow(): Boolean {
        return this is ListeningForFlashCommand || 
               this is ParsingFlashCommand
    }

    /**
     * Lấy tên state để logging
     */
    fun getName(): String {
        return when (this) {
            is Idle -> "Idle"
            is Success -> "Success"
            is Cancel -> "Cancel"
            is Error -> "Error: $errorMessage"
            is ListeningForSMSCommand -> "ListeningForSMSCommand"
            is ParsingSMSCommand -> "ParsingSMSCommand"
            is ConfirmingSMSCommand -> "ConfirmingSMSCommand($receiver, $message)"
            is WaitingForUserConfirmation -> "WaitingForUserConfirmation"
            is SearchingContact -> "SearchingContact($contactName)"
            is SuggestingSimilarContacts -> "SuggestingSimilarContacts(${similarContacts.size} contacts, retry=$retryCount)"
            is WaitingForNewContactName -> "WaitingForNewContactName"
            is SendingSMS -> "SendingSMS($contactName)"
            is ListeningForWACommand -> "ListeningForWACommand"
            is ParsingWACommand -> "ParsingWACommand"
            is ConfirmingWACommand -> "ConfirmingWACommand($receiver, $message)"
            is WaitingForWAUserConfirmation -> "WaitingForWAUserConfirmation"
            is SearchingWAContact -> "SearchingWAContact($contactName)"
            is SuggestingWASimilarContacts -> "SuggestingWASimilarContacts(${similarContacts.size} contacts, retry=$retryCount)"
            is WaitingForWANewContactName -> "WaitingForWANewContactName"
            is SendingWA -> "SendingWA($contactName)"
            is ExecutingWACommand -> "ExecutingWACommand"
            is ListeningForCallCommand -> "ListeningForCallCommand"
            is ParsingCallCommand -> "ParsingCallCommand"
            is MakingCall -> "MakingCall($phoneNumber)"
            is ListeningForChromeCommand -> "ListeningForChromeCommand"
            is ParsingChromeCommand -> "ParsingChromeCommand"
            is SearchingChrome -> "SearchingChrome($query)"
            is ListeningForYouTubeCommand -> "ListeningForYouTubeCommand"
            is ParsingYouTubeCommand -> "ParsingYouTubeCommand"
            is SearchingYouTube -> "SearchingYouTube($query)"
            is ListeningForAlarmCommand -> "ListeningForAlarmCommand"
            is ParsingAlarmCommand -> "ParsingAlarmCommand"
            is ExecutingAlarmCommand -> "ExecutingAlarmCommand"
            is ExecutingChromeCommand -> "ExecutingChromeCommand"
            is ExecutingYouTubeCommand -> "ExecutingYouTubeCommand"
        is ExecutingSMSCommand -> "ExecutingSMSCommand"
        is ConfirmingPhoneCommand -> "ConfirmingPhoneCommand"
        is ExecutingPhoneCommand -> "ExecutingPhoneCommand"
            is CreatingAlarm -> "CreatingAlarm($hour:$minute - $message)"
            is ListeningForCalendarCommand -> "ListeningForCalendarCommand"
            is ParsingCalendarCommand -> "ParsingCalendarCommand"
            is CreatingCalendarEvent -> "CreatingCalendarEvent($title)"
            
            // Device Control States
            is ListeningForCameraCommand -> "ListeningForCameraCommand"
            is ParsingCameraCommand -> "ParsingCameraCommand"
            is ExecutingCameraCommand -> "ExecutingCameraCommand"
            is ListeningForWifiCommand -> "ListeningForWifiCommand"
            is ParsingWifiCommand -> "ParsingWifiCommand"
            is ExecutingWifiCommand -> "ExecutingWifiCommand"
            is ListeningForVolumeCommand -> "ListeningForVolumeCommand"
            is ParsingVolumeCommand -> "ParsingVolumeCommand"
            is ExecutingVolumeCommand -> "ExecutingVolumeCommand"
            is ListeningForFlashCommand -> "ListeningForFlashCommand"
            is ParsingFlashCommand -> "ParsingFlashCommand"
            is ExecutingFlashCommand -> "ExecutingFlashCommand"
            is AskingContactName -> "AskingContactName"
            is AskingContactPhone -> "AskingContactPhone($contactName)"
            is ExecutingAddContactCommand -> "ExecutingAddContactCommand"
        }
    }
}