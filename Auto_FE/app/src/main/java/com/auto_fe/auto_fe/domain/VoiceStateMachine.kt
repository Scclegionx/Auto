package com.auto_fe.auto_fe.domain

import android.util.Log

/**
 * Base State Machine Engine
 * Quản lý việc chuyển đổi states và xử lý side effects
 */
abstract class VoiceStateMachine {

    companion object {
        private const val TAG = "VoiceStateMachine"
    }

    // Current state
    var currentState: VoiceState = VoiceState.Idle
        private set

    // State change listener
    var onStateChanged: ((VoiceState, VoiceState) -> Unit)? = null

    // Event listener (for debugging)
    var onEventProcessed: ((VoiceEvent) -> Unit)? = null

    /**
     * Process an event và chuyển đổi state
     * @param event Event cần xử lý
     * @return true nếu event được xử lý thành công, false nếu không
     */
    fun processEvent(event: VoiceEvent): Boolean {
        val oldState = currentState

        Log.d(TAG, "Processing event: ${event.getName()}")
        Log.d(TAG, "Current state: ${currentState.getName()}")

        // Callback để notify event được xử lý
        onEventProcessed?.invoke(event)

        // Get new state from transition logic
        val newState = getNextState(currentState, event)

        if (newState == null) {
            Log.w(TAG, "No transition defined for event ${event.getName()} in state ${currentState.getName()}")
            return false
        }

        // State changed
        if (newState != currentState) {
            Log.d(TAG, "State transition: ${oldState.getName()} -> ${newState.getName()}")

            // Update current state
            currentState = newState

            // Notify listener
            onStateChanged?.invoke(oldState, newState)

            // Execute side effects khi enter new state
            onEnterState(newState, event)
        } else {
            Log.d(TAG, "State unchanged: ${currentState.getName()}")
        }

        return true
    }

    /**
     * Abstract method: Định nghĩa transition logic
     * Subclass phải implement method này để define các transitions
     *
     * @param currentState State hiện tại
     * @param event Event trigger transition
     * @return New state sau khi transition, hoặc null nếu không có transition
     */
    protected abstract fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState?

    /**
     * Abstract method: Side effects khi enter new state
     * Subclass có thể override để thực hiện actions khi enter state
     *
     * @param state State vừa enter
     * @param event Event trigger transition
     */
    protected abstract fun onEnterState(state: VoiceState, event: VoiceEvent)

    /**
     * Reset state machine về trạng thái ban đầu
     */
    fun reset() {
        Log.d(TAG, "Resetting state machine")
        val oldState = currentState
        currentState = VoiceState.Idle
        onStateChanged?.invoke(oldState, currentState)
    }

    /**
     * Kiểm tra xem có thể process event này không
     * @param event Event cần kiểm tra
     * @return true nếu event có thể được xử lý ở state hiện tại
     */
    fun canProcessEvent(event: VoiceEvent): Boolean {
        return getNextState(currentState, event) != null
    }

    /**
     * Get current state name (for debugging)
     */
    fun getCurrentStateName(): String {
        return currentState.getName()
    }

    /**
     * Kiểm tra xem state machine đã ở terminal state chưa
     */
    fun isTerminal(): Boolean {
        return currentState.isTerminal()
    }
}