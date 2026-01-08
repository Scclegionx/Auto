package com.auto_fe.auto_fe.ui.components.popup

import android.content.Context
import android.graphics.PixelFormat
import android.util.Log
import android.view.*
import android.widget.ImageView
import com.auto_fe.auto_fe.R
import com.auto_fe.auto_fe.base.callback.CommandProcessorCallback
import com.auto_fe.auto_fe.core.CommandProcessor
import kotlinx.coroutines.*

class FloatingWindow(private val context: Context) {

    // 1. Quản lý Window & UI
    private var windowManager: WindowManager? = null
    private var floatingView: View? = null
    private var layoutParams: WindowManager.LayoutParams? = null

    // Popup hiển thị sóng âm
    private var voiceRecordingPopup: LiquidWaveRecordingPopup? = null

    // 2. Logic Core
    // Tạo Scope riêng cho Service, dùng SupervisorJob để lỗi con không làm chết Service
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    // CommandProcessor để xử lý giọng nói
    private val commandProcessor = CommandProcessor(context, scope)

    // 3. States
    private var isRecording = false
    private var confirmationQuestion = ""
    private var successMessage = ""
    private var errorMessage = ""

    // Drag functionality variables
    private var initialX = 0
    private var initialY = 0
    private var initialTouchX = 0f
    private var initialTouchY = 0f

    init {
        // Khởi tạo Popup Visualizer
        voiceRecordingPopup = LiquidWaveRecordingPopup(context)

        // Setup popup callbacks
        setupPopupCallbacks()
    }

    private fun setupPopupCallbacks() {
        // Nếu người dùng bấm nút STOP trên cái Popup sóng nước to
        voiceRecordingPopup?.onStopClick = {
            Log.d("FloatingWindow", "Popup stop clicked")
            handleMicAction() // Gọi hàm xử lý chính để dừng
        }
    }

    /**
     * Hàm xử lý chính: Toggle Ghi âm / Hủy
     * Logic đồng bộ với VoiceScreen
     */
    private fun handleMicAction() {
        if (!isRecording) {
            // --- BẮT ĐẦU GHI ÂM ---
            Log.d("FloatingWindow", "Starting voice flow")
            isRecording = true
            updateFloatingButtonIcon()

            // 1. Hiển thị Popup sóng âm
            voiceRecordingPopup?.show()

            // 2. Gọi CommandProcessor chạy Workflow
            confirmationQuestion = ""
            successMessage = ""
            errorMessage = ""
            
            commandProcessor.startVoiceControl(object : CommandProcessorCallback {
                override fun onCommandExecuted(success: Boolean, message: String) {
                    Log.d("FloatingWindow", "Success: $message")
                    isRecording = false
                    confirmationQuestion = ""
                    successMessage = message
                    errorMessage = ""
                    voiceRecordingPopup?.updateStatus("")
                    voiceRecordingPopup?.updateConfirmation("")
                    voiceRecordingPopup?.updateSuccess(message)
                    voiceRecordingPopup?.updateError("")
                    // Reset UI sau khi xong
                    scope.launch {
                        delay(3000)
                        successMessage = ""
                        voiceRecordingPopup?.updateSuccess("")
                        resetUI()
                    }
                }

                override fun onError(error: String) {
                    Log.e("FloatingWindow", "Error: $error")
                    isRecording = false
                    errorMessage = error
                    confirmationQuestion = ""
                    successMessage = ""
                    voiceRecordingPopup?.updateStatus("")
                    voiceRecordingPopup?.updateConfirmation("")
                    voiceRecordingPopup?.updateSuccess("")
                    voiceRecordingPopup?.updateError(error)
                    // Reset UI
                    scope.launch {
                        delay(3000)
                        errorMessage = ""
                        voiceRecordingPopup?.updateError("")
                        resetUI()
                    }
                }

                override fun onConfirmationRequired(question: String) {
                    Log.d("FloatingWindow", "Confirmation: $question")
                    confirmationQuestion = question
                    successMessage = ""
                    errorMessage = ""
                    voiceRecordingPopup?.updateStatus("")
                    voiceRecordingPopup?.updateConfirmation(question)
                    voiceRecordingPopup?.updateSuccess("")
                    voiceRecordingPopup?.updateError("")
                }

                override fun onVoiceLevelChanged(level: Int) {
                    // Update sóng âm trên Popup (để nó nhảy nhảy)
                    voiceRecordingPopup?.updateAudioLevel(level)
                }
            })
        } else {
            // --- HỦY BỎ ---
            Log.d("FloatingWindow", "Cancelling voice flow")
            commandProcessor.cancel()
            confirmationQuestion = ""
            successMessage = ""
            errorMessage = ""
            resetUI()
        }
    }

    private fun resetUI() {
        // Delay nhẹ để người dùng kịp đọc thông báo trên popup
        scope.launch {
            delay(1500)
            isRecording = false
            updateFloatingButtonIcon()
            voiceRecordingPopup?.hide()
        }
    }

    fun showFloatingWindow() {
        if (floatingView != null) return

        Log.d("FloatingWindow", "Creating floating window...")
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        floatingView = createCircularFloatingButton()

        // Layout Params cho nút tròn nổi
        // TYPE_APPLICATION_OVERLAY bắt buộc cho Android O trở lên
        layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE, // Không chiếm focus bàn phím
            PixelFormat.TRANSLUCENT
        )

        // Vị trí mặc định
        layoutParams?.gravity = Gravity.TOP or Gravity.START
        layoutParams?.x = 0
        layoutParams?.y = 100

        try {
            windowManager?.addView(floatingView, layoutParams)
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error adding view: ${e.message}")
        }
    }

    private fun createCircularFloatingButton(): View {
        val imageView = ImageView(context)
        imageView.setImageResource(R.drawable.ic_mic_white) // Đảm bảo icon tồn tại
        imageView.setBackgroundResource(R.drawable.selector_circular_button)
        imageView.scaleType = ImageView.ScaleType.CENTER_INSIDE
        imageView.setPadding(16, 16, 16, 16)

        // Kích thước nút (60dp)
        val size = (60 * context.resources.displayMetrics.density).toInt()
        imageView.layoutParams = ViewGroup.LayoutParams(size, size)

        // Click Listener: Gọi vào hàm xử lý chính
        imageView.setOnClickListener {
            handleMicAction()
        }

        // Drag Logic
        setupDragLogic(imageView)

        return imageView
    }

    private fun setupDragLogic(view: View) {
        var isDragging = false
        view.setOnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = layoutParams?.x ?: 0
                    initialY = layoutParams?.y ?: 0
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    isDragging = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val deltaX = (event.rawX - initialTouchX).toInt()
                    val deltaY = (event.rawY - initialTouchY).toInt()

                    // Chỉ coi là Drag nếu di chuyển quá 10px
                    if (Math.abs(deltaX) > 10 || Math.abs(deltaY) > 10) {
                        isDragging = true
                        layoutParams?.x = initialX + deltaX
                        layoutParams?.y = initialY + deltaY
                        windowManager?.updateViewLayout(floatingView, layoutParams)
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (isDragging) {
                        snapToEdge()
                    } else {
                        v.performClick() // Nếu không drag thì kích hoạt Click
                    }
                    true
                }
                else -> false
            }
        }
    }

    private fun updateFloatingButtonIcon() {
        val imageView = floatingView as? ImageView
        val iconRes = if (isRecording) R.drawable.ic_stop_white else R.drawable.ic_mic_white
        imageView?.setImageResource(iconRes)
    }

    private fun snapToEdge() {
        val screenWidth = context.resources.displayMetrics.widthPixels
        layoutParams?.let { params ->
            val centerX = screenWidth / 2
            // Tự động dính vào cạnh trái hoặc phải
            val newX = if (params.x < centerX) 0 else screenWidth - (floatingView?.width ?: 0)
            params.x = newX
            windowManager?.updateViewLayout(floatingView, params)
        }
    }

    fun hideFloatingWindow() {
        // Dừng ghi âm nếu đang chạy
        if (isRecording) commandProcessor.cancel()

        // Ẩn popup sóng nước
        voiceRecordingPopup?.hide()

        // Xóa nút nổi
        try {
            floatingView?.let { windowManager?.removeView(it) }
            floatingView = null
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error removing view", e)
        }
    }

    fun release() {
        hideFloatingWindow()
        // Hủy scope để giải phóng tài nguyên Service
        scope.cancel()
        Log.d("FloatingWindow", "Resources released")
    }
}