package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.third_apps.ChromeAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

/**
 * State Machine cho luồng tìm kiếm Chrome tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> NLP parsing -> tìm kiếm Chrome
 */
class SearchChromeStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val chromeAutomation: ChromeAutomation
) : VoiceStateMachine() {

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "SearchChromeStateMachine"
    }

    // Current search query
    private var currentQuery: String = ""

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartChromeCommand -> VoiceState.ExecutingChromeCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ExecutingChromeCommand -> {
                when (event) {
                    is VoiceEvent.ChromeSearchSuccessfully -> VoiceState.Success
                    is VoiceEvent.ChromeSearchFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.ChromeSearchSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.ChromeSearchFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ExecutingChromeCommand -> {
                Log.d(TAG, "Executing Chrome search command: $currentQuery")
                executeChromeSearch()
            }

            is VoiceState.Success -> {
                speak("Đã mở Chrome tìm kiếm thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.ChromeSearchSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi tìm kiếm Chrome.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.ChromeSearchFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh tìm kiếm Chrome.")
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
     * Được gọi từ CommandProcessor để thực hiện Chrome search command
     */
    fun executeChromeSearchCommand(query: String) {
        Log.d(TAG, "Chrome search command received from CommandProcessor: $query")
        currentQuery = query
        processEvent(VoiceEvent.StartChromeCommand)
    }

    private fun executeChromeSearch() {
        Log.d(TAG, "Executing Chrome search")
        
        chromeAutomation.searchChrome(currentQuery, object : ChromeAutomation.ChromeCallback {
            override fun onSuccess() {
                Log.d(TAG, "Chrome search started successfully")
                processEvent(VoiceEvent.ChromeSearchSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Chrome search failed: $error")
                processEvent(VoiceEvent.ChromeSearchFailed(error))
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
