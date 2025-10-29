package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.automation.device.CameraAutomation
import com.auto_fe.auto_fe.audio.VoiceManager
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

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "CameraStateMachine"
    }

    // Current command data
    private var currentAction: String = "" // "photo" or "video"

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartCameraCapture -> VoiceState.ExecutingCameraCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ExecutingCameraCommand -> {
                when (event) {
                    is VoiceEvent.CameraCapturedSuccessfully -> VoiceState.Success
                    is VoiceEvent.CameraCaptureFailed -> VoiceState.Error(event.error)
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
            is VoiceState.ExecutingCameraCommand -> {
                Log.d(TAG, "Executing camera command: $currentAction")
                when (currentAction) {
                    "photo" -> executePhotoCapture()
                    "video" -> executeVideoCapture()
                    else -> {
                        Log.e(TAG, "Unknown camera action: $currentAction")
                        processEvent(VoiceEvent.CameraCaptureFailed("Lệnh camera không được hỗ trợ"))
                    }
                }
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

    /**
     * Được gọi từ CommandProcessor để thực hiện camera command
     */
    fun executeCameraCommand(action: String) {
        Log.d(TAG, "Camera command received from CommandProcessor: $action")
        currentAction = action
        processEvent(VoiceEvent.StartCameraCapture)
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

    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    private fun resetContext() {
        currentAction = ""
    }

    fun cleanup() {
        // Cleanup resources if needed
    }
}
