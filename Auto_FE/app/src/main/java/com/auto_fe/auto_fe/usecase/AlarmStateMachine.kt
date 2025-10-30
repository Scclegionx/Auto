package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.alarm.AlarmAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

/**
 * State Machine cho luồng tạo báo thức tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> NLP parsing -> tạo báo thức
 */
class AlarmStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val alarmAutomation: AlarmAutomation
) : VoiceStateMachine() {

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "AlarmStateMachine"
    }

    // Current alarm data
    private var currentHour: Int = 0
    private var currentMinute: Int = 0
    private var currentMessage: String = ""
    private var currentYear: Int? = null
    private var currentMonth: Int? = null // 1-12
    private var currentDay: Int? = null

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartAlarmCommand -> VoiceState.ExecutingAlarmCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ExecutingAlarmCommand -> {
                when (event) {
                    is VoiceEvent.AlarmCreatedSuccessfully -> VoiceState.Success
                    is VoiceEvent.AlarmCreationFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.AlarmCreatedSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.AlarmCreationFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ExecutingAlarmCommand -> {
                Log.d(TAG, "Executing alarm command: ${currentHour}:${currentMinute} - $currentMessage")
                executeAlarmCreation()
            }

            is VoiceState.Success -> {
                speak("Đã tạo báo thức thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.AlarmCreatedSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi tạo báo thức.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.AlarmCreationFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh tạo báo thức.")
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
     * Được gọi từ CommandProcessor để thực hiện alarm command
     */
    fun executeAlarmCommand(hour: Int, minute: Int, message: String) {
        Log.d(TAG, "Alarm command received from CommandProcessor: $hour:$minute - $message")
        currentHour = hour
        currentMinute = minute
        currentMessage = message
        processEvent(VoiceEvent.StartAlarmCommand)
    }

    /**
     * Được gọi từ CommandProcessor để thực hiện alarm command với NGÀY cụ thể
     */
    fun executeAlarmCommandOnDate(year: Int, month: Int, day: Int, hour: Int, minute: Int, message: String) {
        Log.d(TAG, "Alarm command (with date) received: $day/$month/$year $hour:$minute - $message")
        currentYear = year
        currentMonth = month
        currentDay = day
        currentHour = hour
        currentMinute = minute
        currentMessage = message
        processEvent(VoiceEvent.StartAlarmCommand)
    }

    private fun executeAlarmCreation() {
        Log.d(TAG, "Executing alarm creation")
        val y = currentYear
        val m = currentMonth
        val d = currentDay
        if (y != null && m != null && d != null) {
            alarmAutomation.createAlarmOnDate(y, m, d, currentHour, currentMinute, currentMessage, object : AlarmAutomation.AlarmCallback {
                override fun onSuccess() {
                    Log.d(TAG, "Alarm created successfully (with date)")
                    processEvent(VoiceEvent.AlarmCreatedSuccessfully)
                }

                override fun onError(error: String) {
                    Log.e(TAG, "Alarm creation failed: $error")
                    processEvent(VoiceEvent.AlarmCreationFailed(error))
                }
            })
            return
        }

        alarmAutomation.createAlarm(currentHour, currentMinute, emptyList(), currentMessage, object : AlarmAutomation.AlarmCallback {
            override fun onSuccess() {
                Log.d(TAG, "Alarm created successfully")
                processEvent(VoiceEvent.AlarmCreatedSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Alarm creation failed: $error")
                processEvent(VoiceEvent.AlarmCreationFailed(error))
            }
        })
    }

    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    private fun resetContext() {
        currentHour = 0
        currentMinute = 0
        currentMessage = ""
        currentYear = null
        currentMonth = null
        currentDay = null
    }

    fun cleanup() {
        // Cleanup resources if needed
    }
}
