package com.auto_fe.auto_fe.ui

import android.content.Context
import android.graphics.PixelFormat
import android.view.Gravity
import android.view.LayoutInflater
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.Toast
import android.util.Log
import android.app.AlertDialog
import android.view.animation.AnimationUtils
import com.auto_fe.auto_fe.R
import com.auto_fe.auto_fe.audio.AudioManager
import com.auto_fe.auto_fe.core.CommandProcessor
import com.auto_fe.auto_fe.automation.msg.SMSAutomation

class FloatingWindow(private val context: Context) {
    private var windowManager: WindowManager? = null
    private var floatingView: LinearLayout? = null
    private var popupWindow: PopupWindow? = null
    private var audioManager: AudioManager? = null
    private var commandProcessor: CommandProcessor? = null

    init {
        audioManager = AudioManager(context)
        commandProcessor = CommandProcessor(context)
    }

    fun showFloatingWindow() {
        if (floatingView != null) return

        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager

        // Tạo button nhỏ thay vì menu
        floatingView = createFloatingButton()

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        )

        params.gravity = Gravity.TOP or Gravity.START
        params.x = 0
        params.y = 100

        windowManager?.addView(floatingView, params)
    }

    private fun createFloatingButton(): LinearLayout {
        val button = Button(context)
        button.text = "Auto FE"
        button.setBackgroundColor(0xFF4CAF50.toInt()) // Xanh nhạt như ban đầu
        button.setTextColor(0xFFFFFFFF.toInt())
        button.setOnClickListener {
            showCommandMenu()
        }

        val container = LinearLayout(context)
        container.addView(button)
        return container
    }

    private fun showCommandMenu() {
        if (popupWindow != null) return // Đã mở rồi thì không mở lại

        val inflater = LayoutInflater.from(context)
        val menuView = inflater.inflate(R.layout.floating_menu, null)

        val recordButton = menuView.findViewById<Button>(R.id.btn_record)
        val testSmsButton = menuView.findViewById<Button>(R.id.btn_test_sms)
        val testNlpButton = menuView.findViewById<Button>(R.id.btn_test_nlp)
        val closeButton = menuView.findViewById<Button>(R.id.btn_close)

        recordButton.setOnClickListener {
            hideCommandMenu()
            startAudioRecording()
        }

        testSmsButton.setOnClickListener {
            hideCommandMenu()
            testSMSFunction()
        }

        testNlpButton.setOnClickListener {
            hideCommandMenu()
            testNLPFlow()
        }

        closeButton.setOnClickListener {
            hideCommandMenu()
        }

        // Tạo popup window với animation
        popupWindow = PopupWindow(
            menuView,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            true
        )

        // Thiết lập animation (bỏ qua style mặc định)

        // Tự động đóng khi click outside
        popupWindow?.isOutsideTouchable = true
        popupWindow?.isFocusable = true

        // Thêm animation tùy chỉnh
        menuView.startAnimation(AnimationUtils.loadAnimation(context, com.auto_fe.auto_fe.R.anim.popup_enter))

        popupWindow?.showAsDropDown(floatingView)
    }

    private fun hideCommandMenu() {
        popupWindow?.let { popup ->
            // Thêm animation khi đóng
            val menuView = popup.contentView
            menuView?.startAnimation(AnimationUtils.loadAnimation(context, com.auto_fe.auto_fe.R.anim.popup_exit))

            // Đóng popup sau khi animation hoàn thành
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                popup.dismiss()
            }, 150)
        }
        popupWindow = null
    }

    private fun startAudioRecording() {
        audioManager?.startVoiceInteraction(object : AudioManager.AudioManagerCallback {
            override fun onSpeechResult(spokenText: String) {
                audioManager?.confirmCommand(spokenText, object : AudioManager.AudioManagerCallback {
                    override fun onSpeechResult(spokenText: String) {}
                    override fun onConfirmationResult(confirmed: Boolean) {
                        if (confirmed) {
                            commandProcessor?.processCommand(spokenText, object : CommandProcessor.CommandProcessorCallback {
                                override fun onCommandExecuted(success: Boolean, message: String) {
                                    Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                                }
                                override fun onError(error: String) {
                                    Toast.makeText(context, error, Toast.LENGTH_LONG).show()
                                }
                            })
                        } else {
                            Toast.makeText(context, "Lệnh đã bị hủy", Toast.LENGTH_SHORT).show()
                        }
                    }
                    override fun onError(error: String) {
                        Toast.makeText(context, error, Toast.LENGTH_LONG).show()
                    }
                })
            }
            override fun onConfirmationResult(confirmed: Boolean) {}
            override fun onError(error: String) {
                Toast.makeText(context, error, Toast.LENGTH_LONG).show()
            }
        })
    }

    private fun testSMSFunction() {
        Log.d("FloatingWindow", "Starting SMS test...")
        val smsAutomation = SMSAutomation(context)
        smsAutomation.sendSMS("mom", "con sắp về", object : SMSAutomation.SMSCallback {
            override fun onSuccess() {
                Log.d("FloatingWindow", "SMS test successful!")
                showTestResult("Test SMS thành công!")
            }
            override fun onError(error: String) {
                Log.e("FloatingWindow", "SMS test error: $error")
                showTestResult("Test SMS lỗi: $error")
            }
        })
    }

    private fun testNLPFlow() {
        Log.d("FloatingWindow", "Starting NLP flow test...")
        val testCommand = "nhắn tin cho mom là con sắp về"
        Log.d("FloatingWindow", "Testing command: $testCommand")

        commandProcessor?.processCommand(testCommand, object : CommandProcessor.CommandProcessorCallback {
            override fun onCommandExecuted(success: Boolean, message: String) {
                Log.d("FloatingWindow", "NLP flow test result: $success - $message")
                showTestResult("NLP Flow Test: $message")
            }
            override fun onError(error: String) {
                Log.e("FloatingWindow", "NLP flow test error: $error")
                showTestResult("NLP Flow Test Error: $error")
            }
        })
    }

    private fun showTestResult(message: String) {
        try {
            val builder = AlertDialog.Builder(context)
            builder.setTitle("Test Result")
            builder.setMessage(message)
            builder.setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
            builder.show()
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error showing dialog: ${e.message}")
        }
    }

    fun hideFloatingWindow() {
        floatingView?.let { view ->
            windowManager?.removeView(view)
        }
        floatingView = null
        windowManager = null
        audioManager?.release()
    }

    /**
     * Giải phóng tất cả resources để tránh memory leak
     */
    fun release() {
        try {
            // Cleanup CommandProcessor (bao gồm PhoneAutomation TTS)
            commandProcessor?.release()
            commandProcessor = null

            // Cleanup AudioManager
            audioManager?.release()
            audioManager = null

            Log.d("FloatingWindow", "All resources released successfully")
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error releasing resources: ${e.message}")
        }
    }
}