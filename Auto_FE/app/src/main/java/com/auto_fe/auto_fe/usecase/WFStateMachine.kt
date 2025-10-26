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

class WFStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val controlDeviceAutomation: ControlDeviceAutomation
) : VoiceStateMachine() {

    private val commandProcessor = CommandProcessor(context)
    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "WFStateMachine"
        private const val NLP_TIMEOUT_MS = 15000L
    }

    // Current command data
    private var currentCommand: String = ""
    private var currentAction: String = "" // "enable", "disable", "toggle"

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.ListeningForWifiCommand -> {
                when (event) {
                    is VoiceEvent.WifiCommandReceived -> VoiceState.ParsingWifiCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ParsingWifiCommand -> {
                when (event) {
                    is VoiceEvent.WifiCommandParsed -> VoiceState.Success
                    is VoiceEvent.WifiCommandParseFailed -> VoiceState.Error()
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.WifiToggledSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.WifiToggleFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ListeningForWifiCommand -> {
                voiceManager.resetBusyState()
                speakAndListen("Bạn cần tôi trợ giúp điều gì?")
            }

            is VoiceState.ParsingWifiCommand -> {
                speak("Đang xử lý lệnh WiFi...")
                parseCommandAsync(currentCommand)
            }

            is VoiceState.Success -> {
                speak("Đã thực hiện lệnh WiFi thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.WifiToggledSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi thực hiện lệnh WiFi.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.WifiToggleFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh WiFi.")
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
            is VoiceState.ListeningForWifiCommand -> {
                Log.d(TAG, "WiFi command received: $spokenText")
                currentCommand = spokenText
                processEvent(VoiceEvent.WifiCommandReceived(spokenText))
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
                    Log.e(TAG, "NLP timeout for WiFi command")
                    processEvent(VoiceEvent.WifiCommandParseFailed("Timeout: Không nhận được phản hồi từ server"))
                }

                commandProcessor.processCommand(command, object : CommandProcessor.CommandProcessorCallback {
                    override fun onCommandExecuted(success: Boolean, message: String) {
                        timeoutJob.cancel()
                        if (success) {
                            Log.d(TAG, "WiFi command executed successfully: $message")
                            processEvent(VoiceEvent.WifiToggledSuccessfully)
                        } else {
                            Log.e(TAG, "WiFi command execution failed: $message")
                            processEvent(VoiceEvent.WifiCommandParseFailed(message))
                        }
                    }

                    override fun onError(error: String) {
                        timeoutJob.cancel()
                        Log.e(TAG, "NLP error for WiFi command: $error")
                        processEvent(VoiceEvent.WifiCommandParseFailed(error))
                    }

                    override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                        timeoutJob.cancel()
                        Log.d(TAG, "WiFi command needs confirmation: $command, action: $receiver")
                        
                        // receiver contains the action type (enable/disable/toggle)
                        currentAction = receiver
                        
                        when (command) {
                            "wifi-enable" -> {
                                executeEnableWifi()
                            }
                            "wifi-disable" -> {
                                executeDisableWifi()
                            }
                            "wifi-toggle" -> {
                                executeToggleWifi()
                            }
                            else -> {
                                Log.e(TAG, "Unknown WiFi command: $command")
                                processEvent(VoiceEvent.WifiCommandParseFailed("Lệnh WiFi không được hỗ trợ"))
                            }
                        }
                    }
                })
            } catch (e: Exception) {
                Log.e(TAG, "Error processing WiFi command: ${e.message}", e)
                processEvent(VoiceEvent.WifiCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
            }
        }
    }

    private fun executeEnableWifi() {
        Log.d(TAG, "Executing enable WiFi")
        
        controlDeviceAutomation.enableWifi(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "WiFi enabled successfully")
                processEvent(VoiceEvent.WifiToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "WiFi enable failed: $error")
                processEvent(VoiceEvent.WifiToggleFailed(error))
            }
        })
    }

    private fun executeDisableWifi() {
        Log.d(TAG, "Executing disable WiFi")
        
        controlDeviceAutomation.disableWifi(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "WiFi disabled successfully")
                processEvent(VoiceEvent.WifiToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "WiFi disable failed: $error")
                processEvent(VoiceEvent.WifiToggleFailed(error))
            }
        })
    }

    private fun executeToggleWifi() {
        Log.d(TAG, "Executing toggle WiFi")
        
        controlDeviceAutomation.toggleWifi(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "WiFi toggled successfully")
                processEvent(VoiceEvent.WifiToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "WiFi toggle failed: $error")
                processEvent(VoiceEvent.WifiToggleFailed(error))
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
                    processEvent(VoiceEvent.WifiCommandParseFailed("Không nhận được lệnh"))
                }
            }

            override fun onConfirmationResult(confirmed: Boolean) {
                // Not used in this context
            }

            override fun onError(error: String) {
                Log.e(TAG, "Speech recognition error: $error")
                processEvent(VoiceEvent.WifiCommandParseFailed("Lỗi nhận dạng giọng nói: $error"))
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