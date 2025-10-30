package com.auto_fe.auto_fe.core

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.automation.msg.SMSAutomation
import com.auto_fe.auto_fe.automation.msg.WAAutomation
import com.auto_fe.auto_fe.automation.phone.PhoneAutomation
import com.auto_fe.auto_fe.automation.device.CameraAutomation
import com.auto_fe.auto_fe.automation.device.ControlDeviceAutomation
import com.auto_fe.auto_fe.automation.alarm.AlarmAutomation
import com.auto_fe.auto_fe.automation.third_apps.ChromeAutomation
import com.auto_fe.auto_fe.automation.third_apps.YouTubeAutomation
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.service.NLPService
import com.auto_fe.auto_fe.usecase.SendSMSStateMachine
import com.auto_fe.auto_fe.usecase.SendWAStateMachine
import com.auto_fe.auto_fe.usecase.PhoneStateMachine
import com.auto_fe.auto_fe.usecase.CameraStateMachine
import com.auto_fe.auto_fe.usecase.FlashStateMachine
import com.auto_fe.auto_fe.usecase.WFStateMachine
import com.auto_fe.auto_fe.usecase.VolumnStateMachine
import com.auto_fe.auto_fe.usecase.AlarmStateMachine
import com.auto_fe.auto_fe.usecase.SearchChromeStateMachine
import com.auto_fe.auto_fe.usecase.AddContactStateMachine
import com.auto_fe.auto_fe.automation.phone.ContactAutomation
import com.auto_fe.auto_fe.usecase.YoutubeStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject

class CommandProcessor(private val context: Context) {
    private val smsAutomation = SMSAutomation(context)
    private val waAutomation = WAAutomation(context)
    private val phoneAutomation = PhoneAutomation(context)
    private val cameraAutomation = CameraAutomation(context)
    private val controlDeviceAutomation = ControlDeviceAutomation(context)
    private val alarmAutomation = AlarmAutomation(context)
    private val chromeAutomation = ChromeAutomation(context)
    private val youtubeAutomation = YouTubeAutomation(context)
    private val contactAutomation = ContactAutomation(context)
    private val voiceManager = VoiceManager.getInstance(context)
    private val nlpService = NLPService(context)
    
    // State Machines - lazy initialization để tránh tạo nhiều instance
    private val smsStateMachine by lazy { SendSMSStateMachine(context, voiceManager, smsAutomation) }
    private val waStateMachine by lazy { SendWAStateMachine(context, voiceManager, waAutomation) }
    private val phoneStateMachine by lazy { PhoneStateMachine(context, voiceManager, phoneAutomation) }
    private val cameraStateMachine by lazy { CameraStateMachine(context, voiceManager, cameraAutomation) }
    private val flashStateMachine by lazy { FlashStateMachine(context, voiceManager, controlDeviceAutomation) }
    private val wifiStateMachine by lazy { WFStateMachine(context, voiceManager, controlDeviceAutomation) }
    private val volumeStateMachine by lazy { VolumnStateMachine(context, voiceManager, controlDeviceAutomation) }
    private val alarmStateMachine by lazy { AlarmStateMachine(context, voiceManager, alarmAutomation) }
    private val searchChromeStateMachine by lazy { SearchChromeStateMachine(context, voiceManager, chromeAutomation) }
    private val youtubeStateMachine by lazy { YoutubeStateMachine(context, voiceManager, youtubeAutomation) }
    private val addContactStateMachine by lazy { AddContactStateMachine(context, voiceManager, contactAutomation) }

    interface CommandProcessorCallback {
        fun onCommandExecuted(success: Boolean, message: String)
        fun onError(error: String)
        fun onNeedConfirmation(command: String, receiver: String, message: String)
    }
    
    // Callbacks cho StateMachine state changes
    private var onStateSuccess: ((String) -> Unit)? = null
    private var onStateError: ((String) -> Unit)? = null

    /**
     * Setup callbacks để lắng nghe state changes từ StateMachines
     */
    fun setupStateCallbacks(
        onSuccess: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        this.onStateSuccess = onSuccess
        this.onStateError = onError
        
        // Setup callbacks cho tất cả StateMachines
        setupStateMachineCallbacks()
    }
    
    private fun setupStateMachineCallbacks() {
        // Setup SMS StateMachine callbacks
        smsStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã gửi tin nhắn thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Cancel -> {
                    // Cancel state cũng cần dừng ghi âm
                    onStateSuccess?.invoke("Đã hủy lệnh gọi điện.")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }
        
        // Setup WA StateMachine callbacks
        waStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã gửi tin nhắn WhatsApp thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Cancel -> {
                    // Cancel state cũng cần dừng ghi âm
                    onStateSuccess?.invoke("Đã hủy lệnh gửi WhatsApp.")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }
        
        // Setup Phone StateMachine callbacks
        phoneStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã gọi điện thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Cancel -> {
                    // Cancel state cũng cần dừng ghi âm
                    onStateSuccess?.invoke("Đã hủy lệnh gọi điện.")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }
        
        // Setup Camera StateMachine callbacks
        cameraStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã thực hiện lệnh camera thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }
        
        // Setup Flash StateMachine callbacks
        flashStateMachine.onStateChanged = { oldState, newState ->
            Log.d("CommandProcessor", "FlashStateMachine state changed: $oldState -> $newState")
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    Log.d("CommandProcessor", "FlashStateMachine success callback triggered")
                    onStateSuccess?.invoke("Đã thực hiện lệnh đèn flash thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    Log.d("CommandProcessor", "FlashStateMachine error callback triggered")
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }
        
        // Setup WiFi StateMachine callbacks
        wifiStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã thực hiện lệnh WiFi thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }
        
        // Setup Volume StateMachine callbacks
        volumeStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã thực hiện lệnh âm lượng thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }
        
        // Setup Alarm StateMachine callbacks
        alarmStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã tạo báo thức thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }

        // Setup SearchChrome StateMachine callbacks
        searchChromeStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã mở Chrome tìm kiếm thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }

        // Setup AddContact StateMachine callbacks
        addContactStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã thêm liên hệ thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Cancel -> {
                    onStateSuccess?.invoke("Đã hủy thêm liên hệ.")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // No-op
                }
            }
        }
        // Setup YouTube StateMachine callbacks
        youtubeStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã mở YouTube tìm kiếm thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }

        // Setup SMS StateMachine callbacks
        smsStateMachine.onStateChanged = { oldState, newState ->
            when (newState) {
                is com.auto_fe.auto_fe.domain.VoiceState.Success -> {
                    onStateSuccess?.invoke("Đã gửi tin nhắn thành công!")
                }
                is com.auto_fe.auto_fe.domain.VoiceState.Error -> {
                    onStateError?.invoke(newState.errorMessage)
                }
                else -> {
                    // Không cần xử lý các state khác
                }
            }
        }
    }

    /**
     * Giải phóng tất cả resources
     */
    fun release() {
        try {
            phoneAutomation.release()
            smsStateMachine.cleanup()
            waStateMachine.cleanup()
            phoneStateMachine.cleanup()
            cameraStateMachine.cleanup()
            flashStateMachine.cleanup()
            wifiStateMachine.cleanup()
            volumeStateMachine.cleanup()
            alarmStateMachine.cleanup()
            searchChromeStateMachine.cleanup()
            youtubeStateMachine.cleanup()
            Log.d("CommandProcessor", "Resources released successfully")
        } catch (e: Exception) {
            Log.e("CommandProcessor", "Error releasing resources: ${e.message}")
        }
    }

    fun processCommand(command: String, callback: CommandProcessorCallback) {
        Log.d("CommandProcessor", "Processing command: $command")
        CoroutineScope(Dispatchers.Main).launch {
            nlpService.sendCommandToServer(command, object : NLPService.NLPServiceCallback {
                override fun onSuccess(response: NLPService.NLPResponse) {
                    Log.d("CommandProcessor", "NLP Response: ${response.rawJson}")
                    // Xử lý JSON response với entities
                    processCommandResponse(response.rawJson, callback)
                }

                override fun onError(error: String) {
                    Log.e("CommandProcessor", "NLP Error: $error")
                    callback.onError("Lỗi NLP: $error")
                }
            })
        }
    }

    /**
     * Xử lý JSON response với entities
     * {
            "text": "Nhắn tin cho cháu Dương là chiều nay 3h qua nhà bà ăn cơm",
            "intent": "send-mess",
            "confidence": 1.0,
            "command": "Gửi tin nhắn",
            "entities": {
                "RECEIVER": "cháu dương",
                "MESSAGE": "chiều nay 3h qua nhà bà ăn cơm",
                "PLATFORM": "sms"
            },
            "value": "chiều nay 3h qua nhà bà ăn cơm",
            "method": "reasoning_engine",
            "processing_time": 0.301513,
            "timestamp": "2025-10-05T11:07:19.436163",
            "reasoning_details": {}
        }
     */
    private fun processCommandResponse(jsonResponse: String, callback: CommandProcessorCallback) {
        try {
            val json = JSONObject(jsonResponse)
            val command = json.optString("command", "")
            val entities = json.optJSONObject("entities")

            if (entities == null) {
                callback.onError("Thiếu thông tin entities")
                return
            }

            val receiver = entities.optString("RECEIVER", "")
            val platform = entities.optString("PLATFORM", "")
            val message = entities.optString("MESSAGE", "")
            val query = entities.optString("QUERY", "")
            val time = entities.optString("TIME", "")
            val date = entities.optString("DATE", "")
            val device = entities.optString("DEVICE", "")
            val action = entities.optString("ACTION", "")
            val mode = entities.optString("MODE", "")

            when (command) {
                "send-msg" -> {
                    // Dựa vào platform để quyết định gọi SMS hay WhatsApp
                    when (platform.lowercase()) {
                        "sms" -> {
                            Log.d("CommandProcessor", "Routing to SendSMSStateMachine: $receiver -> $message")
                            smsStateMachine.executeSMSCommand(receiver, message)
                        }
                        "whatsapp", "wa" -> {
                            Log.d("CommandProcessor", "Routing to SendWAStateMachine: $receiver -> $message")
                            waStateMachine.executeWACommand(receiver, message)
                        }
                        else -> {
                            Log.d("CommandProcessor", "Unknown platform: $platform, defaulting to SMS")
                            smsStateMachine.executeSMSCommand(receiver, message)
                        }
                    }
                }
                "call" -> {
                    // Parse thành công, gọi trực tiếp PhoneStateMachine
                    Log.d("CommandProcessor", "Routing to PhoneStateMachine: $receiver, platform: $platform")
                    phoneStateMachine.executePhoneCommand(receiver, platform)
                }
                "search-internet" -> {
                    Log.d("CommandProcessor", "Routing to SearchChromeStateMachine for query: $query")
                    searchChromeStateMachine.executeChromeSearchCommand(query)
                }
                "search-youtube" -> {
                    Log.d("CommandProcessor", "Routing to YoutubeStateMachine for query: $query")
                    youtubeStateMachine.executeYouTubeSearchCommand(query)
                }
                "set-alarm" -> {
                    val timeData = parseTimeFromString(time)
                    if (timeData == null) {
                        Log.e("CommandProcessor", "Cannot parse time from: $time")
                        callback.onError("Tôi không hiểu thời gian báo thức. Vui lòng nói rõ hơn, ví dụ: 'Đặt báo thức 9 giờ 30'.")
                        return
                    }

                    val (hour, minute) = timeData
                    val message = "Báo thức"

                    // Nếu có DATE (YYYY-MM-DD), parse và chuyển sang StateMachine với ngày cụ thể
                    if (date.isNotEmpty()) {
                        val dateData = parseDateIso(date)
                        if (dateData != null) {
                            val (year, month, day) = dateData
                            Log.d("CommandProcessor", "Routing to AlarmStateMachine with date: $day/$month/$year $hour:$minute")
                            alarmStateMachine.executeAlarmCommandOnDate(year, month, day, hour, minute, message)
                            return
                        } else {
                            Log.w("CommandProcessor", "DATE provided but invalid: $date. Falling back to time-only alarm.")
                        }
                    }

                    Log.d("CommandProcessor", "Routing to AlarmStateMachine for time-only alarm: $hour:$minute")
                    alarmStateMachine.executeAlarmCommand(hour, minute, message)
                }
                "add-calendar" -> {
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("add-calendar", receiver, message)
                }
                "add-contacts" -> {
                    Log.d("CommandProcessor", "Routing to AddContactStateMachine")
                    addContactStateMachine.startAddContactFlow()
                }
                
                // ========== DEVICE CONTROL COMMANDS ==========
                
                // Camera commands
                "open-cam" -> {
                    val cameraType = mode.lowercase()
                    Log.d("CommandProcessor", "Routing to CameraStateMachine for camera: $cameraType")
                    when (cameraType) {
                        "image" -> {
                            cameraStateMachine.executeCameraCommand("photo")
                        }
                        "video" -> {
                            cameraStateMachine.executeCameraCommand("video")
                        }
                        else -> {
                            Log.e("CommandProcessor", "Unknown camera type: $cameraType")
                            callback.onError("Loại camera không được hỗ trợ: $cameraType")
                        }
                    }
                }
                
                // Control device commands (wifi, volume, flash)
                "control-device" -> {
                    Log.d("CommandProcessor", "Routing control-device: device=$device, action=$action")
                    
                    when (device) {
                        "wifi" -> {
                            wifiStateMachine.executeWifiCommand(action)
                        }
                        "volumn", "volume" -> {
                            volumeStateMachine.executeVolumeCommand(action)
                        }
                        "flash" -> {
                            flashStateMachine.executeFlashCommand(action)
                        }
                        else -> {
                            Log.e("CommandProcessor", "Unknown device: $device")
                            callback.onError("Thiết bị không được hỗ trợ: $device")
                        }
                    }
                }
                
                else -> {
                    callback.onError("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ.")
                }
            }
        } catch (e: Exception) {
            callback.onError("Lỗi xử lý lệnh: ${e.message}")
        }
    }

    
    /**
     * Parse time từ string (ví dụ: "9:00", "9h00", "9 giờ 00")
     */
    private fun parseTimeFromString(timeString: String): Pair<Int, Int>? {
        try {
            val cleanTime = timeString.replace("giờ", ":").replace("h", ":").trim()
            
            // Tìm pattern HH:MM hoặc H:MM
            val timePattern = Regex("(\\d{1,2}):(\\d{2})")
            val match = timePattern.find(cleanTime)
            
            if (match != null) {
                val hour = match.groupValues[1].toInt()
                val minute = match.groupValues[2].toInt()
                
                if (hour in 0..23 && minute in 0..59) {
                    return Pair(hour, minute)
                }
            }
            
            // Fallback: tìm số đầu tiên làm giờ, số thứ hai làm phút
            val numbers = Regex("\\d+").findAll(cleanTime).map { it.value.toInt() }.toList()
            if (numbers.size >= 2) {
                val hour = numbers[0]
                val minute = numbers[1]
                if (hour in 0..23 && minute in 0..59) {
                    return Pair(hour, minute)
                }
            }
            
            return null
        } catch (e: Exception) {
            Log.e("CommandProcessor", "Error parsing time: ${e.message}")
            return null
        }
    }
    
    /**
     * Parse DATE theo định dạng ISO "YYYY-MM-DD" và trả về (year, month, day)
     * month là 1..12
     */
    private fun parseDateIso(dateString: String): Triple<Int, Int, Int>? {
        return try {
            val parts = dateString.trim().split("-")
            if (parts.size != 3) return null
            val year = parts[0].toInt()
            val month = parts[1].toInt()
            val day = parts[2].toInt()
            if (year in 1970..3000 && month in 1..12 && day in 1..31) {
                Triple(year, month, day)
            } else null
        } catch (e: Exception) {
            Log.e("CommandProcessor", "Error parsing date: ${e.message}")
            null
        }
    }
}