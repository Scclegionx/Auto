package com.auto_fe.auto_fe.base.callback

interface CommandProcessorCallback {
    fun onCommandExecuted(success: Boolean, message: String)
    fun onError(error: String)
    fun onVoiceLevelChanged(level: Int)
    fun onConfirmationRequired(question: String) {
        // Default: không làm gì (cho backward compatibility)
    }
}
