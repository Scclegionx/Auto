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
                    "Bạn cần tôi trợ giúp điều gì?"
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
                        else -> {
                            // TODO: Handle other states
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
                if (currentWorkflow?.getCurrentState() is AutomationState.Success || 
                    currentWorkflow?.getCurrentState() is AutomationState.Error) {
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