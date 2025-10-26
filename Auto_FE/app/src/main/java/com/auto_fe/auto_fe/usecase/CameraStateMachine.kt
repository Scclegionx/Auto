package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.automation.device.CameraAutomation
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.core.CommandProcessor
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

class CameraStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val cameraAutomation: CameraAutomation
) : VoiceStateMachine() {

    private val commandProcessor = CommandProcessor(context)
    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "CameraStateMachine"
        private const val NLP_TIMEOUT_MS = 15000L
    }

    // Current command data
    private var currentCommand: String = ""
    private var currentAction: String = "" // "photo" or "video"

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.ListeningForCameraCommand -> {
                when (event) {
                    is VoiceEvent.CameraCommandReceived -> VoiceState.ParsingCameraCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ParsingCameraCommand -> {
                when (event) {
                    is VoiceEvent.CameraCommandParsed -> VoiceState.Success
                    is VoiceEvent.CameraCommandParseFailed -> VoiceState.Error()
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.CameraCapturedSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.CameraCaptureFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ListeningForCameraCommand -> {
                voiceManager.resetBusyState()
                speakAndListen("Bạn cần tôi trợ giúp điều gì?")
            }

            is VoiceState.ParsingCameraCommand -> {
                speak("Đang xử lý lệnh camera...")
                parseCommandAsync(currentCommand)
            }

            is VoiceState.Success -> {
                speak("Đã thực hiện lệnh camera thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.CameraCapturedSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi thực hiện lệnh camera.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.CameraCaptureFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh camera.")
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
            is VoiceState.ListeningForCameraCommand -> {
                Log.d(TAG, "Camera command received: $spokenText")
                currentCommand = spokenText
                processEvent(VoiceEvent.CameraCommandReceived(spokenText))
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
                    Log.e(TAG, "NLP timeout for camera command")
                    processEvent(VoiceEvent.CameraCommandParseFailed("Timeout: Không nhận được phản hồi từ server"))
                }

                commandProcessor.processCommand(command, object : CommandProcessor.CommandProcessorCallback {
                    override fun onCommandExecuted(success: Boolean, message: String) {
                        timeoutJob.cancel()
                        if (success) {
                            Log.d(TAG, "Camera command executed successfully: $message")
                            processEvent(VoiceEvent.CameraCapturedSuccessfully)
                        } else {
                            Log.e(TAG, "Camera command execution failed: $message")
                            processEvent(VoiceEvent.CameraCommandParseFailed(message))
                        }
                    }

                    override fun onError(error: String) {
                        timeoutJob.cancel()
                        Log.e(TAG, "NLP error for camera command: $error")
                        processEvent(VoiceEvent.CameraCommandParseFailed(error))
                    }

                    override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                        timeoutJob.cancel()
                        Log.d(TAG, "Camera command needs confirmation: $command, action: $receiver")
                        
                        // receiver contains the action type (photo/video)
                        currentAction = receiver
                        
                        when (command) {
                            "capture-photo" -> {
                                executePhotoCapture()
                            }
                            "capture-video" -> {
                                executeVideoCapture()
                            }
                            else -> {
                                Log.e(TAG, "Unknown camera command: $command")
                                processEvent(VoiceEvent.CameraCommandParseFailed("Lệnh camera không được hỗ trợ"))
                            }
                        }
                    }
                })
            } catch (e: Exception) {
                Log.e(TAG, "Error processing camera command: ${e.message}", e)
                processEvent(VoiceEvent.CameraCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
            }
        }
    }

    private fun executePhotoCapture() {
        Log.d(TAG, "Executing photo capture")
        
        cameraAutomation.capturePhoto(object : CameraAutomation.CameraCallback {
            override fun onSuccess() {
                Log.d(TAG, "Photo capture started successfully")
                processEvent(VoiceEvent.CameraCapturedSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Photo capture failed: $error")
                processEvent(VoiceEvent.CameraCaptureFailed(error))
            }
        })
    }

    private fun executeVideoCapture() {
        Log.d(TAG, "Executing video capture")
        
        cameraAutomation.captureVideo(object : CameraAutomation.CameraCallback {
            override fun onSuccess() {
                Log.d(TAG, "Video capture started successfully")
                processEvent(VoiceEvent.CameraCapturedSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Video capture failed: $error")
                processEvent(VoiceEvent.CameraCaptureFailed(error))
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
                    processEvent(VoiceEvent.CameraCommandParseFailed("Không nhận được lệnh"))
                }
            }

            override fun onConfirmationResult(confirmed: Boolean) {
                // Not used in this context
            }

            override fun onError(error: String) {
                Log.e(TAG, "Speech recognition error: $error")
                processEvent(VoiceEvent.CameraCommandParseFailed("Lỗi nhận dạng giọng nói: $error"))
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
