package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.alarm.AlarmAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import com.auto_fe.auto_fe.core.CommandProcessor

/**
 * State Machine cho luồng tạo báo thức tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> NLP parsing -> tạo báo thức
 */
class AlarmStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val alarmAutomation: AlarmAutomation
) : VoiceStateMachine() {
    
    private val commandProcessor = CommandProcessor(context)

    companion object {
        private const val TAG = "AlarmStateMachine"
    }

    // Lưu context data trong quá trình xử lý
    private var currentHour: Int = 0
    private var currentMinute: Int = 0
    private var currentMessage: String = ""
    
    // Callback cho transcript
    var onTranscriptUpdated: ((String) -> Unit)? = null

    /**
     * Định nghĩa State Transitions
     */
    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {

            // IDLE -> Start listening
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartRecording -> VoiceState.ListeningForAlarmCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle // Already idle
                    else -> null
                }
            }

            // LISTENING -> Parse command
            is VoiceState.ListeningForAlarmCommand -> {
                when (event) {
                    is VoiceEvent.AlarmCommandReceived -> {
                        VoiceState.ParsingAlarmCommand
                    }
                    is VoiceEvent.SpeechRecognitionFailed -> {
                        VoiceState.Error("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ.")
                    }
                    is VoiceEvent.Timeout -> {
                        VoiceState.Error("Tôi không nhận được phản hồi từ bạn.")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // PARSING -> Create alarm
            is VoiceState.ParsingAlarmCommand -> {
                when (event) {
                    is VoiceEvent.AlarmCommandParsed -> {
                        // Lưu data và chuyển sang tạo báo thức
                        currentHour = event.hour
                        currentMinute = event.minute
                        currentMessage = event.message
                        VoiceState.CreatingAlarm(event.hour, event.minute, event.message)
                    }
                    is VoiceEvent.AlarmCommandParseFailed -> {
                        VoiceState.Error(event.reason)
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // CREATING ALARM -> Handle result
            is VoiceState.CreatingAlarm -> {
                when (event) {
                    is VoiceEvent.AlarmCreatedSuccessfully -> {
                        VoiceState.Success
                    }
                    is VoiceEvent.AlarmCreationFailed -> {
                        VoiceState.Error("Tạo báo thức thất bại: ${event.error}")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // Terminal states - no more transitions
            is VoiceState.Success, is VoiceState.Error -> null

            else -> null
        }
    }

    /**
     * Side Effects khi enter state
     */
    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        Log.d(TAG, "onEnterState: ${state.getName()}")

        when (state) {
            is VoiceState.ListeningForAlarmCommand -> {
                // Reset VoiceManager trước khi bắt đầu
                voiceManager.resetBusyState()
                // Phát câu hỏi và bắt đầu lắng nghe
                speakAndListen("Bạn cần tôi trợ giúp điều gì?")
            }

            is VoiceState.ParsingAlarmCommand -> {
                // Parse command từ event
                if (event is VoiceEvent.AlarmCommandReceived) {
                    parseCommandAsync(event.rawCommand)
                }
            }

            is VoiceState.CreatingAlarm -> {
                // Tạo báo thức
                createAlarmAsync(state.hour, state.minute, state.message)
            }

            is VoiceState.Success -> {
                speak("Đã tạo báo thức thành công.")
            }

            is VoiceState.Error -> {
                speak(state.errorMessage)
            }

            is VoiceState.Idle -> {
                // Reset state when cancelled
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy ghi âm")
                    // Reset all context data
                    currentHour = 0
                    currentMinute = 0
                    currentMessage = ""
                    // Reset VoiceManager busy state
                    voiceManager.resetBusyState()
                }
            }

            else -> {
                // Do nothing
            }
        }
    }

    // ========== HELPER METHODS ==========

    /**
     * Phát giọng nói và bắt đầu lắng nghe
     */
    private fun speakAndListen(text: String, delaySeconds: Int = 2) {
        // Sử dụng API mới với delay tùy chỉnh
        voiceManager.textToSpeech(text, delaySeconds, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                handleSpeechResult(spokenText)
            }

            override fun onConfirmationResult(confirmed: Boolean) {}

            override fun onError(error: String) {
                processEvent(VoiceEvent.SpeechRecognitionFailed)
            }

            override fun onAudioLevelChanged(level: Int) {
                // No audio level visualization for alarm creation
            }
        })
    }

    /**
     * Chỉ phát giọng nói
     */
    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    /**
     * Xử lý kết quả speech recognition dựa vào state hiện tại
     */
    private fun handleSpeechResult(spokenText: String) {
        try {
            Log.d(TAG, "Handling speech result: $spokenText")
            
            // Update transcript in popup
            onTranscriptUpdated?.invoke(spokenText)
            
            when (currentState) {
                is VoiceState.ListeningForAlarmCommand -> {
                    processEvent(VoiceEvent.AlarmCommandReceived(spokenText))
                }

                else -> {
                    Log.w(TAG, "Unexpected speech result in state: ${currentState.getName()}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in handleSpeechResult: ${e.message}")
            processEvent(VoiceEvent.SpeechRecognitionFailed)
        }
    }

    /**
     * Parse alarm command using NLP service - chỉ gửi raw command và nhận JSON response
     */
    private fun parseCommandAsync(command: String) {
        Log.d(TAG, "Sending command to NLP: $command")
        
        try {
            // Add timeout protection
            val timeoutHandler = android.os.Handler(android.os.Looper.getMainLooper())
            var isCallbackExecuted = false
            
            val timeoutRunnable = Runnable {
                if (!isCallbackExecuted) {
                    Log.e(TAG, "NLP service timeout")
                    isCallbackExecuted = true
                    processEvent(VoiceEvent.AlarmCommandParseFailed("Lỗi hệ thống: Không nhận được phản hồi từ NLP service."))
                }
            }
            
            // Set timeout of 10 seconds
            timeoutHandler.postDelayed(timeoutRunnable, 10000)
            
            commandProcessor.processCommand(command, object : CommandProcessor.CommandProcessorCallback {
                override fun onCommandExecuted(success: Boolean, message: String) {
                    if (!isCallbackExecuted) {
                        isCallbackExecuted = true
                        timeoutHandler.removeCallbacks(timeoutRunnable)
                        
                        try {
                            if (success) {
                                Log.d(TAG, "NLP processed successfully: $message")
                                // NLP đã xử lý xong và tạo báo thức, chuyển sang Success
                                processEvent(VoiceEvent.AlarmCreatedSuccessfully)
                            } else {
                                Log.e(TAG, "NLP processing failed: $message")
                                processEvent(VoiceEvent.AlarmCommandParseFailed(message))
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onCommandExecuted: ${e.message}")
                            processEvent(VoiceEvent.AlarmCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
                        }
                    }
                }

                override fun onError(error: String) {
                    if (!isCallbackExecuted) {
                        isCallbackExecuted = true
                        timeoutHandler.removeCallbacks(timeoutRunnable)
                        
                        try {
                            Log.e(TAG, "NLP Error: $error")
                            // Phân biệt lỗi hệ thống vs lỗi không hỗ trợ lệnh
                            if (error.contains("Lỗi NLP") || error.contains("Server error") || error.contains("Connection") || 
                                error.contains("Lỗi kết nối") || error.contains("Lỗi không xác định")) {
                                processEvent(VoiceEvent.AlarmCommandParseFailed("Lỗi hệ thống: $error"))
                            } else {
                                processEvent(VoiceEvent.AlarmCommandParseFailed("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ."))
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onError: ${e.message}")
                            processEvent(VoiceEvent.AlarmCommandParseFailed("Lỗi xử lý: ${e.message}"))
                        }
                    }
                }

                override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                    if (!isCallbackExecuted) {
                        isCallbackExecuted = true
                        timeoutHandler.removeCallbacks(timeoutRunnable)
                        
                        try {
                            Log.d(TAG, "NLP needs confirmation: $command -> time: $receiver, message: $message")
                            // Parse time từ receiver (ví dụ: "9:30" hoặc "9h30")
                            val timeData = parseTimeFromString(receiver)
                            if (timeData != null) {
                                currentHour = timeData.first
                                currentMinute = timeData.second
                                currentMessage = message.ifEmpty { "Báo thức" }
                                // Parse thành công, chuyển sang tạo báo thức
                                processEvent(VoiceEvent.AlarmCommandParsed(currentHour, currentMinute, currentMessage))
                            } else {
                                processEvent(VoiceEvent.AlarmCommandParseFailed("Tôi không hiểu thời gian báo thức. Vui lòng nói rõ hơn, ví dụ: 'Tạo báo thức 9 giờ 30'."))
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onNeedConfirmation: ${e.message}")
                            processEvent(VoiceEvent.AlarmCommandParseFailed("Lỗi xác nhận: ${e.message}"))
                        }
                    }
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Error in parseCommandAsync: ${e.message}")
            processEvent(VoiceEvent.AlarmCommandParseFailed("Lỗi phân tích lệnh: ${e.message}"))
        }
    }

    /**
     * Parse time từ string (ví dụ: "9:30", "9h30", "9 giờ 30")
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
            Log.e(TAG, "Error parsing time: ${e.message}")
            return null
        }
    }

    /**
     * Tạo báo thức
     */
    private fun createAlarmAsync(hour: Int, minute: Int, message: String) {
        alarmAutomation.createAlarm(hour, minute, emptyList(), message, object : AlarmAutomation.AlarmCallback {
            override fun onSuccess() {
                processEvent(VoiceEvent.AlarmCreatedSuccessfully)
            }

            override fun onError(error: String) {
                processEvent(VoiceEvent.AlarmCreationFailed(error))
            }
        })
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        reset()
        currentHour = 0
        currentMinute = 0
        currentMessage = ""
        commandProcessor.release()
    }
}
