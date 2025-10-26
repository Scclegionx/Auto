package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.phone.PhoneAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import com.auto_fe.auto_fe.core.CommandProcessor

/**
 * State Machine cho luồng gọi điện tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> NLP parsing -> gọi điện
 */
class PhoneStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val phoneAutomation: PhoneAutomation
) : VoiceStateMachine() {
    
    private val commandProcessor = CommandProcessor(context)

    companion object {
        private const val TAG = "PhoneStateMachine"
    }

    // Lưu context data trong quá trình xử lý
    private var currentContact: String = ""
    
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
                    is VoiceEvent.StartRecording -> VoiceState.ListeningForCallCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle // Already idle
                    else -> null
                }
            }

            // LISTENING -> Parse command
            is VoiceState.ListeningForCallCommand -> {
                when (event) {
                    is VoiceEvent.CallCommandReceived -> {
                        VoiceState.ParsingCallCommand
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

            // PARSING -> Make call
            is VoiceState.ParsingCallCommand -> {
                when (event) {
                    is VoiceEvent.CallCommandParsed -> {
                        // Lưu data và chuyển sang gọi điện
                        currentContact = event.contactName
                        VoiceState.MakingCall(event.contactName)
                    }
                    is VoiceEvent.CallCommandParseFailed -> {
                        VoiceState.Error(event.reason)
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // MAKING CALL -> Handle result
            is VoiceState.MakingCall -> {
                when (event) {
                    is VoiceEvent.CallMadeSuccessfully -> {
                        VoiceState.Success
                    }
                    is VoiceEvent.CallFailed -> {
                        VoiceState.Error("Gọi điện thất bại: ${event.error}")
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
            is VoiceState.ListeningForCallCommand -> {
                // Reset VoiceManager trước khi bắt đầu
                voiceManager.resetBusyState()
                // Phát câu hỏi và bắt đầu lắng nghe
                speakAndListen("Bạn cần tôi trợ giúp điều gì?")
            }

            is VoiceState.ParsingCallCommand -> {
                // Parse command từ event
                if (event is VoiceEvent.CallCommandReceived) {
                    parseCommandAsync(event.rawCommand)
                }
            }

            is VoiceState.MakingCall -> {
                // Gọi điện
                makeCallAsync(state.phoneNumber)
            }

            is VoiceState.Success -> {
                speak("Đã gọi điện.")
            }

            is VoiceState.Error -> {
                speak(state.errorMessage)
            }

            is VoiceState.Idle -> {
                // Reset state when cancelled
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy ghi âm")
                    // Reset all context data
                    currentContact = ""
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
                // No audio level visualization for phone calls
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
                is VoiceState.ListeningForCallCommand -> {
                    processEvent(VoiceEvent.CallCommandReceived(spokenText))
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
     * Parse call command using NLP service - chỉ gửi raw command và nhận JSON response
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
                    processEvent(VoiceEvent.CallCommandParseFailed("Lỗi hệ thống: Không nhận được phản hồi từ NLP service."))
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
                                // NLP đã xử lý xong và gọi điện, chuyển sang Success
                                processEvent(VoiceEvent.CallMadeSuccessfully)
                            } else {
                                Log.e(TAG, "NLP processing failed: $message")
                                processEvent(VoiceEvent.CallCommandParseFailed(message))
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onCommandExecuted: ${e.message}")
                            processEvent(VoiceEvent.CallCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
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
                                processEvent(VoiceEvent.CallCommandParseFailed("Lỗi hệ thống: $error"))
                            } else {
                                processEvent(VoiceEvent.CallCommandParseFailed("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ."))
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onError: ${e.message}")
                            processEvent(VoiceEvent.CallCommandParseFailed("Lỗi xử lý: ${e.message}"))
                        }
                    }
                }

                override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                    if (!isCallbackExecuted) {
                        isCallbackExecuted = true
                        timeoutHandler.removeCallbacks(timeoutRunnable)
                        
                        try {
                            Log.d(TAG, "NLP needs confirmation: $command -> $receiver")
                            // Lưu thông tin từ NLP response để gọi điện
                            currentContact = receiver
                            // Parse thành công, chuyển sang gọi điện
                            processEvent(VoiceEvent.CallCommandParsed(receiver))
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onNeedConfirmation: ${e.message}")
                            processEvent(VoiceEvent.CallCommandParseFailed("Lỗi xác nhận: ${e.message}"))
                        }
                    }
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Error in parseCommandAsync: ${e.message}")
            processEvent(VoiceEvent.CallCommandParseFailed("Lỗi phân tích lệnh: ${e.message}"))
        }
    }

    /**
     * Gọi điện
     */
    private fun makeCallAsync(contactName: String) {
        phoneAutomation.makeCall(contactName, object : PhoneAutomation.PhoneCallback {
            override fun onSuccess() {
                processEvent(VoiceEvent.CallMadeSuccessfully)
            }

            override fun onError(error: String) {
                processEvent(VoiceEvent.CallFailed(error))
            }
        })
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        reset()
        currentContact = ""
        commandProcessor.release()
        phoneAutomation.release()
    }
}
