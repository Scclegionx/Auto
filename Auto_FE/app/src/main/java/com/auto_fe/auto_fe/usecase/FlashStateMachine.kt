package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.automation.device.ControlDeviceAutomation
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.core.CommandProcessor
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

class FlashStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val controlDeviceAutomation: ControlDeviceAutomation
) : VoiceStateMachine() {

    private val commandProcessor = CommandProcessor(context)
    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "FlashStateMachine"
        private const val NLP_TIMEOUT_MS = 15000L
    }

    // Current command data
    private var currentCommand: String = ""
    private var currentAction: String = "" // "enable", "disable", "toggle"

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.ListeningForFlashCommand -> {
                when (event) {
                    is VoiceEvent.FlashCommandReceived -> VoiceState.ParsingFlashCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ParsingFlashCommand -> {
                when (event) {
                    is VoiceEvent.FlashCommandParsed -> VoiceState.Success
                    is VoiceEvent.FlashCommandParseFailed -> VoiceState.Error()
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.FlashToggledSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.FlashToggleFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ListeningForFlashCommand -> {
                voiceManager.resetBusyState()
                speakAndListen("Bạn cần tôi trợ giúp điều gì?")
            }

            is VoiceState.ParsingFlashCommand -> {
                speak("Đang xử lý lệnh đèn flash...")
                parseCommandAsync(currentCommand)
            }

            is VoiceState.Success -> {
                speak("Đã thực hiện lệnh đèn flash thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.FlashToggledSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi thực hiện lệnh đèn flash.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.FlashToggleFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh đèn flash.")
                }
                voiceManager.resetBusyState()
                resetContext()
            }
            else -> {
                // Handle other states if needed
            }
        }
    }

    fun handleSpeechResult(spokenText: String) {
        when (currentState) {
            is VoiceState.ListeningForFlashCommand -> {
                Log.d(TAG, "Flash command received: $spokenText")
                currentCommand = spokenText
                processEvent(VoiceEvent.FlashCommandReceived(spokenText))
            }
            else -> {
                // Handle other states if needed
            }
        }
    }

    private fun parseCommandAsync(command: String) {
        coroutineScope.launch {
            try {
                // Timeout protection
                val timeoutJob = launch {
                    delay(NLP_TIMEOUT_MS)
                    Log.e(TAG, "NLP timeout for flash command")
                    processEvent(VoiceEvent.FlashCommandParseFailed("Timeout: Không nhận được phản hồi từ server"))
                }

                commandProcessor.processCommand(command, object : CommandProcessor.CommandProcessorCallback {
                    override fun onCommandExecuted(success: Boolean, message: String) {
                        timeoutJob.cancel()
                        if (success) {
                            Log.d(TAG, "Flash command executed successfully: $message")
                            processEvent(VoiceEvent.FlashToggledSuccessfully)
                        } else {
                            Log.e(TAG, "Flash command execution failed: $message")
                            processEvent(VoiceEvent.FlashCommandParseFailed(message))
                        }
                    }

                    override fun onError(error: String) {
                        timeoutJob.cancel()
                        Log.e(TAG, "NLP error for flash command: $error")
                        processEvent(VoiceEvent.FlashCommandParseFailed(error))
                    }

                    override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                        timeoutJob.cancel()
                        Log.d(TAG, "Flash command needs confirmation: $command, action: $receiver")
                        
                        // receiver contains the action type (enable/disable/toggle)
                        currentAction = receiver
                        
                        when (command) {
                            "flash-enable" -> {
                                executeEnableFlash()
                            }
                            "flash-disable" -> {
                                executeDisableFlash()
                            }
                            "flash-toggle" -> {
                                executeToggleFlash()
                            }
                            else -> {
                                Log.e(TAG, "Unknown flash command: $command")
                                processEvent(VoiceEvent.FlashCommandParseFailed("Lệnh đèn flash không được hỗ trợ"))
                            }
                        }
                    }
                })
            } catch (e: Exception) {
                Log.e(TAG, "Error processing flash command: ${e.message}", e)
                processEvent(VoiceEvent.FlashCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
            }
        }
    }

    private fun executeEnableFlash() {
        Log.d(TAG, "Executing enable flash")
        
        controlDeviceAutomation.enableFlash(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Flash enabled successfully")
                processEvent(VoiceEvent.FlashToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Flash enable failed: $error")
                processEvent(VoiceEvent.FlashToggleFailed(error))
            }
        })
    }

    private fun executeDisableFlash() {
        Log.d(TAG, "Executing disable flash")
        
        controlDeviceAutomation.disableFlash(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Flash disabled successfully")
                processEvent(VoiceEvent.FlashToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Flash disable failed: $error")
                processEvent(VoiceEvent.FlashToggleFailed(error))
            }
        })
    }

    private fun executeToggleFlash() {
        Log.d(TAG, "Executing toggle flash")
        
        controlDeviceAutomation.toggleFlash(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Flash toggled successfully")
                processEvent(VoiceEvent.FlashToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Flash toggle failed: $error")
                processEvent(VoiceEvent.FlashToggleFailed(error))
            }
        })
    }

    private fun speakAndListen(text: String) {
        voiceManager.textToSpeech(text, 0, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                if (spokenText.isNotEmpty()) {
                    handleSpeechResult(spokenText)
                } else {
                    Log.w(TAG, "Empty speech result")
                    processEvent(VoiceEvent.FlashCommandParseFailed("Không nhận được lệnh"))
                }
            }

            override fun onConfirmationResult(confirmed: Boolean) {
                // Not used in this context
            }

            override fun onError(error: String) {
                Log.e(TAG, "Speech recognition error: $error")
                processEvent(VoiceEvent.FlashCommandParseFailed("Lỗi nhận dạng giọng nói: $error"))
            }

            override fun onAudioLevelChanged(level: Int) {
                // Not used in this context
            }
        })
    }

    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    private fun resetContext() {
        currentCommand = ""
        currentAction = ""
    }

    fun cleanup() {
        commandProcessor.release()
    }
}
