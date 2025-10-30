package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.phone.ContactAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class AddContactStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val contactAutomation: ContactAutomation
) : VoiceStateMachine() {

    companion object {
        private const val TAG = "AddContactStateMachine"
    }

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    // Context data
    private var currentName: String = ""
    private var currentPhone: String = ""

    // Optional UI callbacks
    var onAudioLevelChanged: ((Int) -> Unit)? = null
    var onTranscriptUpdated: ((String) -> Unit)? = null

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartAddContactCommand -> VoiceState.AskingContactName
                    else -> null
                }
            }
            is VoiceState.AskingContactName -> {
                when (event) {
                    is VoiceEvent.ContactNameProvided -> VoiceState.AskingContactPhone(event.contactName)
                    is VoiceEvent.ContactAddFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.AskingContactPhone -> {
                when (event) {
                    is VoiceEvent.ContactPhoneProvided -> VoiceState.ExecutingAddContactCommand
                    is VoiceEvent.ContactAddFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ExecutingAddContactCommand -> {
                when (event) {
                    is VoiceEvent.ContactAddedSuccessfully -> VoiceState.Success
                    is VoiceEvent.ContactAddFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.ContactAddedSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.ContactAddFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.AskingContactName -> {
                // Đảm bảo giải phóng busy trước khi hỏi tiếp
                voiceManager.resetBusyState()
                coroutineScope.launch {
                    delay(200)
                    askForContactName()
                }
            }
            is VoiceState.AskingContactPhone -> {
                currentName = state.contactName
                // Đảm bảo giải phóng busy trước khi hỏi tiếp
                voiceManager.resetBusyState()
                coroutineScope.launch {
                    delay(200)
                    askForContactPhone(state.contactName)
                }
            }
            is VoiceState.ExecutingAddContactCommand -> {
                executeAddContact()
            }
            is VoiceState.Success -> {
                speak("Đã thêm liên hệ thành công!")
                voiceManager.resetBusyState()
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.ContactAddedSuccessfully)
                }
            }
            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi thêm liên hệ.")
                voiceManager.resetBusyState()
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.ContactAddFailed("Unknown error"))
                }
            }
            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy thêm liên hệ.")
                }
                voiceManager.resetBusyState()
                resetContext()
            }
            else -> {}
        }
    }

    fun startAddContactFlow() {
        Log.d(TAG, "startAddContactFlow called")
        processEvent(VoiceEvent.StartAddContactCommand)
    }

    private fun askForContactName() {
        val prompt = "Bạn hãy nói tên liên hệ bạn muốn thêm."
        voiceManager.textToSpeech(prompt, 1, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                onTranscriptUpdated?.invoke(spokenText)
                val nameText = spokenText.trim()
                if (nameText.isNotEmpty()) {
                    processEvent(VoiceEvent.ContactNameProvided(nameText))
                } else {
                    speak("Tôi chưa nghe rõ tên. Vui lòng nói lại tên liên hệ.")
                    // hỏi lại
                    askForContactName()
                }
            }
            override fun onConfirmationResult(confirmed: Boolean) {}
            override fun onError(error: String) {
                processEvent(VoiceEvent.ContactAddFailed(error))
            }
            override fun onAudioLevelChanged(level: Int) {
                onAudioLevelChanged?.invoke(level)
            }
        })
    }

    private fun askForContactPhone(name: String) {
        val prompt = "Bạn hãy nói số điện thoại của liên hệ này."
        voiceManager.textToSpeech(prompt, 1, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                onTranscriptUpdated?.invoke(spokenText)
                val rawPhone = spokenText.trim()
                val phone = normalizePhoneDigits(rawPhone)
                if (phone.isNotEmpty()) {
                    currentPhone = phone
                    processEvent(VoiceEvent.ContactPhoneProvided(phone))
                } else {
                    speak("Tôi chưa nghe rõ số điện thoại. Vui lòng nói lại số.")
                    // Reset busy và hỏi lại sau một nhịp ngắn
                    voiceManager.resetBusyState()
                    coroutineScope.launch {
                        delay(200)
                        askForContactPhone(name)
                    }
                }
            }
            override fun onConfirmationResult(confirmed: Boolean) {}
            override fun onError(error: String) {
                // Nếu VoiceManager đang bận, reset và hỏi lại
                if (error.contains("bận", ignoreCase = true)) {
                    voiceManager.resetBusyState()
                    coroutineScope.launch {
                        delay(200)
                        askForContactPhone(name)
                    }
                } else {
                    processEvent(VoiceEvent.ContactAddFailed(error))
                }
            }
            override fun onAudioLevelChanged(level: Int) {
                onAudioLevelChanged?.invoke(level)
            }
        })
    }

    private fun executeAddContact() {
        val name = currentName
        val phone = currentPhone
        Log.d(TAG, "Creating contact: $name - $phone")
        // Ưu tiên chèn trực tiếp (không cần UI). Nếu lỗi (thiếu quyền...), fallback về Intent INSERT.
        contactAutomation.insertContactDirect(name, phone, null, object : ContactAutomation.ContactCallback {
            override fun onSuccess() {
                processEvent(VoiceEvent.ContactAddedSuccessfully)
            }
            override fun onError(error: String) {
                Log.w(TAG, "insertContactDirect failed: $error, falling back to Intent INSERT if possible")
                contactAutomation.insertContact(name, phone, null, object : ContactAutomation.ContactCallback {
                    override fun onSuccess() {
                        processEvent(VoiceEvent.ContactAddedSuccessfully)
                    }
                    override fun onError(error2: String) {
                        processEvent(VoiceEvent.ContactAddFailed(error2))
                    }
                })
            }
        })
    }

    private fun normalizePhoneDigits(input: String): String {
        // Lấy các ký tự số và dấu '+' đầu nếu có
        val digits = input.replace("[^+0-9]".toRegex(), "")
        return digits
    }

    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    private fun resetContext() {
        currentName = ""
        currentPhone = ""
    }
}
