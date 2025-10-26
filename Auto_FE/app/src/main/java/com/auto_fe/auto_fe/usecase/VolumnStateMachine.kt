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

class VolumnStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val controlDeviceAutomation: ControlDeviceAutomation
) : VoiceStateMachine() {

    private val commandProcessor = CommandProcessor(context)
    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "VolumnStateMachine"
        private const val NLP_TIMEOUT_MS = 15000L
    }

    // Current command data
    private var currentCommand: String = ""
    private var currentAction: String = "" // "increase", "decrease", "set"
    private var currentValue: Int = 0 // For set volume

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.ListeningForVolumeCommand -> {
                when (event) {
                    is VoiceEvent.VolumeCommandReceived -> VoiceState.ParsingVolumeCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ParsingVolumeCommand -> {
                when (event) {
                    is VoiceEvent.VolumeCommandParsed -> VoiceState.Success
                    is VoiceEvent.VolumeCommandParseFailed -> VoiceState.Error()
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.VolumeAdjustedSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.VolumeAdjustmentFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ListeningForVolumeCommand -> {
                voiceManager.resetBusyState()
                speakAndListen("Bạn cần tôi trợ giúp điều gì?")
            }

            is VoiceState.ParsingVolumeCommand -> {
                speak("Đang xử lý lệnh âm lượng...")
                parseCommandAsync(currentCommand)
            }

            is VoiceState.Success -> {
                speak("Đã thực hiện lệnh âm lượng thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.VolumeAdjustedSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi thực hiện lệnh âm lượng.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.VolumeAdjustmentFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh âm lượng.")
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
            is VoiceState.ListeningForVolumeCommand -> {
                Log.d(TAG, "Volume command received: $spokenText")
                currentCommand = spokenText
                processEvent(VoiceEvent.VolumeCommandReceived(spokenText))
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
                    Log.e(TAG, "NLP timeout for volume command")
                    processEvent(VoiceEvent.VolumeCommandParseFailed("Timeout: Không nhận được phản hồi từ server"))
                }

                commandProcessor.processCommand(command, object : CommandProcessor.CommandProcessorCallback {
                    override fun onCommandExecuted(success: Boolean, message: String) {
                        timeoutJob.cancel()
                        if (success) {
                            Log.d(TAG, "Volume command executed successfully: $message")
                            processEvent(VoiceEvent.VolumeAdjustedSuccessfully)
                        } else {
                            Log.e(TAG, "Volume command execution failed: $message")
                            processEvent(VoiceEvent.VolumeCommandParseFailed(message))
                        }
                    }

                    override fun onError(error: String) {
                        timeoutJob.cancel()
                        Log.e(TAG, "NLP error for volume command: $error")
                        processEvent(VoiceEvent.VolumeCommandParseFailed(error))
                    }

                    override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                        timeoutJob.cancel()
                        Log.d(TAG, "Volume command needs confirmation: $command, action: $receiver, value: $message")
                        
                        // receiver contains the action type (increase/decrease/set)
                        // message contains the value (for set volume)
                        currentAction = receiver
                        currentValue = message.toIntOrNull() ?: 0
                        
                        when (command) {
                            "volume-increase" -> {
                                executeIncreaseVolume()
                            }
                            "volume-decrease" -> {
                                executeDecreaseVolume()
                            }
                            "volume-set" -> {
                                executeSetVolume(currentValue)
                            }
                            else -> {
                                Log.e(TAG, "Unknown volume command: $command")
                                processEvent(VoiceEvent.VolumeCommandParseFailed("Lệnh âm lượng không được hỗ trợ"))
                            }
                        }
                    }
                })
            } catch (e: Exception) {
                Log.e(TAG, "Error processing volume command: ${e.message}", e)
                processEvent(VoiceEvent.VolumeCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
            }
        }
    }

    private fun executeIncreaseVolume() {
        Log.d(TAG, "Executing increase volume")
        
        controlDeviceAutomation.increaseVolume(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Volume increased successfully")
                processEvent(VoiceEvent.VolumeAdjustedSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Volume increase failed: $error")
                processEvent(VoiceEvent.VolumeAdjustmentFailed(error))
            }
        })
    }

    private fun executeDecreaseVolume() {
        Log.d(TAG, "Executing decrease volume")
        
        controlDeviceAutomation.decreaseVolume(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Volume decreased successfully")
                processEvent(VoiceEvent.VolumeAdjustedSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Volume decrease failed: $error")
                processEvent(VoiceEvent.VolumeAdjustmentFailed(error))
            }
        })
    }

    private fun executeSetVolume(percentage: Int) {
        Log.d(TAG, "Executing set volume to $percentage%")
        
        controlDeviceAutomation.setVolume(percentage, object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Volume set to $percentage% successfully")
                processEvent(VoiceEvent.VolumeAdjustedSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Volume set failed: $error")
                processEvent(VoiceEvent.VolumeAdjustmentFailed(error))
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
                    processEvent(VoiceEvent.VolumeCommandParseFailed("Không nhận được lệnh"))
                }
            }

            override fun onConfirmationResult(confirmed: Boolean) {
                // Not used in this context
            }

            override fun onError(error: String) {
                Log.e(TAG, "Speech recognition error: $error")
                processEvent(VoiceEvent.VolumeCommandParseFailed("Lỗi nhận dạng giọng nói: $error"))
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
        currentValue = 0
    }

    fun cleanup() {
        commandProcessor.release()
    }
}