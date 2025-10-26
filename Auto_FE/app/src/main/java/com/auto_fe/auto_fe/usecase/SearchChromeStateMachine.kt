package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.third_apps.ChromeAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import com.auto_fe.auto_fe.core.CommandProcessor

/**
 * State Machine cho luồng tìm kiếm Chrome tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> NLP parsing -> tìm kiếm Chrome
 */
class SearchChromeStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val chromeAutomation: ChromeAutomation
) : VoiceStateMachine() {
    
    private val commandProcessor = CommandProcessor(context)

    companion object {
        private const val TAG = "SearchChromeStateMachine"
    }

    // Lưu context data trong quá trình xử lý
    private var currentQuery: String = ""
    
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
                    is VoiceEvent.StartRecording -> VoiceState.ListeningForChromeCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle // Already idle
                    else -> null
                }
            }

            // LISTENING -> Parse command
            is VoiceState.ListeningForChromeCommand -> {
                when (event) {
                    is VoiceEvent.ChromeCommandReceived -> {
                        VoiceState.ParsingChromeCommand
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

            // PARSING -> Search Chrome
            is VoiceState.ParsingChromeCommand -> {
                when (event) {
                    is VoiceEvent.ChromeCommandParsed -> {
                        // Lưu data và chuyển sang tìm kiếm
                        currentQuery = event.query
                        VoiceState.SearchingChrome(event.query)
                    }
                    is VoiceEvent.ChromeCommandParseFailed -> {
                        VoiceState.Error(event.reason)
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // SEARCHING CHROME -> Handle result
            is VoiceState.SearchingChrome -> {
                when (event) {
                    is VoiceEvent.ChromeSearchSuccessfully -> {
                        VoiceState.Success
                    }
                    is VoiceEvent.ChromeSearchFailed -> {
                        VoiceState.Error("Tìm kiếm Chrome thất bại: ${event.error}")
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
            is VoiceState.ListeningForChromeCommand -> {
                // Reset VoiceManager trước khi bắt đầu
                voiceManager.resetBusyState()
                // Phát câu hỏi và bắt đầu lắng nghe
                speakAndListen("Bạn cần tôi trợ giúp điều gì?")
            }

            is VoiceState.ParsingChromeCommand -> {
                // Parse command từ event
                if (event is VoiceEvent.ChromeCommandReceived) {
                    parseCommandAsync(event.rawCommand)
                }
            }

            is VoiceState.SearchingChrome -> {
                // Tìm kiếm Chrome
                searchChromeAsync(state.query)
            }

            is VoiceState.Success -> {
                speak("Đã mở Chrome tìm kiếm.")
            }

            is VoiceState.Error -> {
                speak(state.errorMessage)
            }

            is VoiceState.Idle -> {
                // Reset state when cancelled
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy ghi âm")
                    // Reset all context data
                    currentQuery = ""
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
                // No audio level visualization for Chrome search
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
                is VoiceState.ListeningForChromeCommand -> {
                    processEvent(VoiceEvent.ChromeCommandReceived(spokenText))
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
     * Parse Chrome command using NLP service - chỉ gửi raw command và nhận JSON response
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
                    processEvent(VoiceEvent.ChromeCommandParseFailed("Lỗi hệ thống: Không nhận được phản hồi từ NLP service."))
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
                                // NLP đã xử lý xong và tìm kiếm Chrome, chuyển sang Success
                                processEvent(VoiceEvent.ChromeSearchSuccessfully)
                            } else {
                                Log.e(TAG, "NLP processing failed: $message")
                                processEvent(VoiceEvent.ChromeCommandParseFailed(message))
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onCommandExecuted: ${e.message}")
                            processEvent(VoiceEvent.ChromeCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
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
                                processEvent(VoiceEvent.ChromeCommandParseFailed("Lỗi hệ thống: $error"))
                            } else {
                                processEvent(VoiceEvent.ChromeCommandParseFailed("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ."))
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onError: ${e.message}")
                            processEvent(VoiceEvent.ChromeCommandParseFailed("Lỗi xử lý: ${e.message}"))
                        }
                    }
                }

                override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                    if (!isCallbackExecuted) {
                        isCallbackExecuted = true
                        timeoutHandler.removeCallbacks(timeoutRunnable)
                        
                        try {
                            Log.d(TAG, "NLP needs confirmation: $command -> query: $receiver")
                            // Lưu thông tin từ NLP response để tìm kiếm
                            currentQuery = receiver
                            // Parse thành công, chuyển sang tìm kiếm
                            processEvent(VoiceEvent.ChromeCommandParsed(receiver))
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in onNeedConfirmation: ${e.message}")
                            processEvent(VoiceEvent.ChromeCommandParseFailed("Lỗi xác nhận: ${e.message}"))
                        }
                    }
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Error in parseCommandAsync: ${e.message}")
            processEvent(VoiceEvent.ChromeCommandParseFailed("Lỗi phân tích lệnh: ${e.message}"))
        }
    }

    /**
     * Tìm kiếm Chrome
     */
    private fun searchChromeAsync(query: String) {
        chromeAutomation.searchChrome(query, object : ChromeAutomation.ChromeCallback {
            override fun onSuccess() {
                processEvent(VoiceEvent.ChromeSearchSuccessfully)
            }

            override fun onError(error: String) {
                processEvent(VoiceEvent.ChromeSearchFailed(error))
            }
        })
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        reset()
        currentQuery = ""
        commandProcessor.release()
    }
}
