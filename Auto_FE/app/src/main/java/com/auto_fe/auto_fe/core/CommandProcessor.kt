package com.auto_fe.auto_fe.core

import android.content.Context
import com.auto_fe.auto_fe.base.AutomationState
import com.auto_fe.auto_fe.base.callback.CommandProcessorCallback
import com.auto_fe.auto_fe.workflows.AutomationWorkflowManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch

class CommandProcessor(
    private val context: Context,
    private val scope: CoroutineScope
) {
    private var currentWorkflow: AutomationWorkflowManager? = null

    fun startVoiceControl(callback: CommandProcessorCallback) {
        // Chạy một Coroutine trong scope đã được quản lý
        scope.launch {
            try {
                val dispatcher = CommandDispatcher(context)

                val workflow = AutomationWorkflowManager(
                    context,
                    dispatcher,
                    "Dạ, bác cần con giúp điều gì ạ?"
                )

                currentWorkflow = workflow

                workflow.onStateChanged = { state ->
                    when (state) {
                        is AutomationState.Success -> {
                            callback.onCommandExecuted(true, state.message)
                        }
                        is AutomationState.Error -> {
                            callback.onError(state.message)
                        }
                        is AutomationState.Confirmation -> {
                            callback.onConfirmationRequired(state.question)
                        }
                        else -> {
                            // Các state khác (Speaking, Listening, Processing) không cần handle
                        }
                    }
                }

                workflow.onAudioLevelChanged = { level ->
                    callback.onVoiceLevelChanged(level)
                }

                workflow.start()

            } catch (e: Exception) {
                callback.onError("Lỗi hệ thống: ${e.message}")
            } finally {
                // Cleanup khi xong
                val finalState = currentWorkflow?.getCurrentState()
                if (finalState is AutomationState.Success || 
                    finalState is AutomationState.Error) {
                    currentWorkflow = null
                }
            }
        }
    }
    fun cancel() {
        currentWorkflow?.cancel()
        currentWorkflow = null
    }
}