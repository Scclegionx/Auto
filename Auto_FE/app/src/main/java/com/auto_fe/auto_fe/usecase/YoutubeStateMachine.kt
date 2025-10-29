package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.third_apps.YouTubeAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

/**
 * State Machine cho luồng tìm kiếm YouTube tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> tìm kiếm YouTube
 */
class YoutubeStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val youtubeAutomation: YouTubeAutomation
) : VoiceStateMachine() {

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "YoutubeStateMachine"
    }

    // Current search query
    private var currentQuery: String = ""

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartYouTubeCommand -> VoiceState.ExecutingYouTubeCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ExecutingYouTubeCommand -> {
                when (event) {
                    is VoiceEvent.YouTubeSearchSuccessfully -> VoiceState.Success
                    is VoiceEvent.YouTubeSearchFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.YouTubeSearchSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.YouTubeSearchFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ExecutingYouTubeCommand -> {
                Log.d(TAG, "Executing YouTube search command: $currentQuery")
                executeYouTubeSearch()
            }

            is VoiceState.Success -> {
                speak("Đã mở YouTube tìm kiếm thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.YouTubeSearchSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi tìm kiếm YouTube.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.YouTubeSearchFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh tìm kiếm YouTube.")
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
     * Được gọi từ CommandProcessor để thực hiện YouTube search command
     */
    fun executeYouTubeSearchCommand(query: String) {
        Log.d(TAG, "YouTube search command received from CommandProcessor: $query")
        currentQuery = query
        processEvent(VoiceEvent.StartYouTubeCommand)
    }

    private fun executeYouTubeSearch() {
        Log.d(TAG, "Executing YouTube search")
        
        youtubeAutomation.searchYouTube(currentQuery, object : YouTubeAutomation.YouTubeCallback {
            override fun onSuccess() {
                Log.d(TAG, "YouTube search started successfully")
                processEvent(VoiceEvent.YouTubeSearchSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "YouTube search failed: $error")
                processEvent(VoiceEvent.YouTubeSearchFailed(error))
            }
        })
    }

    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    private fun resetContext() {
        currentQuery = ""
    }

    fun cleanup() {
        // Cleanup resources if needed
    }
}
