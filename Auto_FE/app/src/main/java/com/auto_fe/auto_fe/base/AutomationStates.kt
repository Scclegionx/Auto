package com.auto_fe.auto_fe.base

sealed class AutomationState {
    data class Speaking(val text: String) : AutomationState()
    object Listening : AutomationState()
    data class Processing(val rawInput: String) : AutomationState()
    data class Success(val message: String) : AutomationState()
    data class Error(val message: String) : AutomationState()
}

