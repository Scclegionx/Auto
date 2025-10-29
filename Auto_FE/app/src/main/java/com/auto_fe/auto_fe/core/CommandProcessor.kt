package com.auto_fe.auto_fe.core

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.automation.msg.SMSAutomation
import com.auto_fe.auto_fe.automation.phone.PhoneAutomation
import com.auto_fe.auto_fe.automation.device.CameraAutomation
import com.auto_fe.auto_fe.automation.device.ControlDeviceAutomation
import com.auto_fe.auto_fe.automation.alarm.AlarmAutomation
import com.auto_fe.auto_fe.automation.third_apps.ChromeAutomation
import com.auto_fe.auto_fe.automation.third_apps.YouTubeAutomation
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.service.NLPService
import com.auto_fe.auto_fe.usecase.SendSMSStateMachine
import com.auto_fe.auto_fe.usecase.PhoneStateMachine
import com.auto_fe.auto_fe.usecase.CameraStateMachine
import com.auto_fe.auto_fe.usecase.FlashStateMachine
import com.auto_fe.auto_fe.usecase.WFStateMachine
import com.auto_fe.auto_fe.usecase.VolumnStateMachine
import com.auto_fe.auto_fe.usecase.AlarmStateMachine
import com.auto_fe.auto_fe.usecase.SearchChromeStateMachine
import com.auto_fe.auto_fe.usecase.YoutubeStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject

class CommandProcessor(private val context: Context) {
    private val smsAutomation = SMSAutomation(context)
    private val phoneAutomation = PhoneAutomation(context)
    private val cameraAutomation = CameraAutomation(context)
    private val controlDeviceAutomation = ControlDeviceAutomation(context)
    private val alarmAutomation = AlarmAutomation(context)
    private val chromeAutomation = ChromeAutomation(context)
    private val youtubeAutomation = YouTubeAutomation(context)
    private val voiceManager = VoiceManager.getInstance(context)
    private val nlpService = NLPService(context)
    
    // State Machines - lazy initialization để tránh tạo nhiều instance
    private val smsStateMachine by lazy { SendSMSStateMachine(context, voiceManager, smsAutomation) }
    private val phoneStateMachine by lazy { PhoneStateMachine(context, voiceManager, phoneAutomation) }
    private val cameraStateMachine by lazy { CameraStateMachine(context, voiceManager, cameraAutomation) }
    private val flashStateMachine by lazy { FlashStateMachine(context, voiceManager, controlDeviceAutomation) }
    private val wifiStateMachine by lazy { WFStateMachine(context, voiceManager, controlDeviceAutomation) }
    private val volumeStateMachine by lazy { VolumnStateMachine(context, voiceManager, controlDeviceAutomation) }
    private val alarmStateMachine by lazy { AlarmStateMachine(context, voiceManager, alarmAutomation) }
    private val searchChromeStateMachine by lazy { SearchChromeStateMachine(context, voiceManager, chromeAutomation) }
    private val youtubeStateMachine by lazy { YoutubeStateMachine(context, voiceManager, youtubeAutomation) }

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
            val value = json.optString("value", "")

            if (entities == null) {
                callback.onError("Thiếu thông tin entities")
                return
            }

            val receiver = entities.optString("RECEIVER", "")
            val platform = entities.optString("PLATFORM", "")

            when (command) {
                "Gửi tin nhắn" -> {
                    // Parse thành công, gọi trực tiếp SendSMSStateMachine
                    Log.d("CommandProcessor", "Routing to SendSMSStateMachine: $receiver -> $value")
                    smsStateMachine.executeSMSCommand(receiver, value)
                }
                "send-msg" -> {
                    // Parse thành công, gọi trực tiếp SendSMSStateMachine
                    Log.d("CommandProcessor", "Routing to SendSMSStateMachine: $receiver -> $value")
                    smsStateMachine.executeSMSCommand(receiver, value)
                }
                "call" -> {
                    // Parse thành công, gọi trực tiếp PhoneStateMachine
                    Log.d("CommandProcessor", "Routing to PhoneStateMachine: $receiver, platform: $platform")
                    phoneStateMachine.executePhoneCommand(receiver, platform)
                }
                "search" -> {
                    // Xử lý lệnh search dựa vào platform
                    when (platform.lowercase()) {
                        "chrome" -> {
                            Log.d("CommandProcessor", "Routing to SearchChromeStateMachine for query: $value")
                            searchChromeStateMachine.executeChromeSearchCommand(value)
                        }
                        "youtube" -> {
                            // Parse thành công, gọi trực tiếp YoutubeStateMachine
                            Log.d("CommandProcessor", "Routing to YoutubeStateMachine for query: $value")
                            youtubeStateMachine.executeYouTubeSearchCommand(value)
                        }
                        else -> {
                            Log.e("CommandProcessor", "Unsupported search platform: $platform")
                            callback.onError("Tôi không hỗ trợ nền tảng tìm kiếm: $platform")
                        }
                    }
                }
                "search-chrome" -> {
                    // Parse thành công, gọi trực tiếp SearchChromeStateMachine
                    Log.d("CommandProcessor", "Routing to SearchChromeStateMachine for query: $receiver")
                    searchChromeStateMachine.executeChromeSearchCommand(receiver)
                    // StateMachine sẽ tự xử lý, không cần callback
                }
                "search-youtube" -> {
                    // Parse thành công, gọi trực tiếp YoutubeStateMachine
                    Log.d("CommandProcessor", "Routing to YoutubeStateMachine for query: $receiver")
                    youtubeStateMachine.executeYouTubeSearchCommand(receiver)
                    // StateMachine sẽ tự xử lý, không cần callback
                }
                "set-alarm" -> {
                    // Parse time từ value (ví dụ: "9:00")
                    val timeData = parseTimeFromString(value)
                    if (timeData != null) {
                        val hour = timeData.first
                        val minute = timeData.second
                        val message = "Báo thức"
                        Log.d("CommandProcessor", "Routing to AlarmStateMachine for alarm: $hour:$minute")
                        alarmStateMachine.executeAlarmCommand(hour, minute, message)
                        // StateMachine sẽ tự xử lý, không cần callback
                    } else {
                        Log.e("CommandProcessor", "Cannot parse time from: $value")
                        callback.onError("Tôi không hiểu thời gian báo thức. Vui lòng nói rõ hơn, ví dụ: 'Đặt báo thức 9 giờ 30'.")
                    }
                }
                "add-calendar" -> {
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("add-calendar", receiver, value)
                }
                
                // ========== DEVICE CONTROL COMMANDS ==========
                
                // Camera commands
                "open-cam" -> {
                    val cameraType = value.lowercase()
                    Log.d("CommandProcessor", "Routing to CameraStateMachine for camera: $cameraType")
                    when (cameraType) {
                        "image" -> {
                            cameraStateMachine.executeCameraCommand("photo")
                            // StateMachine sẽ tự xử lý, không cần callback
                        }
                        "video" -> {
                            cameraStateMachine.executeCameraCommand("video")
                            // StateMachine sẽ tự xử lý, không cần callback
                        }
                        else -> {
                            Log.e("CommandProcessor", "Unknown camera type: $cameraType")
                            callback.onError("Loại camera không được hỗ trợ: $cameraType")
                        }
                    }
                }
                
                // Control device commands (wifi, volume, flash)
                "control-device" -> {
                    val device = entities.optString("DEVICE", "").lowercase()
                    val action = value.lowercase()
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
     * Xử lý khi user nói lại tên người nhận
     */
    fun processSMSWithNewReceiver(newReceiver: String, message: String, callback: CommandProcessorCallback) {
        smsAutomation.sendSMSWithSmartHandling(newReceiver, message, object : SMSAutomation.SMSConversationCallback {
            override fun onSuccess() {
                callback.onCommandExecuted(true, "Đã gửi tin nhắn thành công")
            }
            override fun onError(error: String) {
                callback.onError(error)
            }
            override fun onNeedConfirmation(similarContacts: List<String>, originalName: String) {
                val contactList = similarContacts.joinToString(" và ")
                val message = "Không tìm thấy danh bạ $originalName nhưng tìm được ${similarContacts.size} danh bạ có tên gần giống là $contactList. Liệu bạn có nhầm lẫn tên người gửi không? Nếu nhầm lẫn bạn hãy nói lại tên"
                callback.onError(message)
            }
        })
    }
}