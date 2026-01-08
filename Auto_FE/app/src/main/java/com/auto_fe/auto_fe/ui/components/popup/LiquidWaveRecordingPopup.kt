package com.auto_fe.auto_fe.ui.components.popup

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.Context
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.auto_fe.auto_fe.R
import kotlin.random.Random

/**
 * Liquid Wave Recording Popup - Visual Upgrade
 * Được đồng bộ hóa với FloatingWindow.kt
 */
class LiquidWaveRecordingPopup(private val context: Context) {

    private var windowManager: WindowManager? = null
    private var popupView: View? = null
    private var isShowing = false

    // UI components
    private var tvTimer: TextView? = null
    private var soundWaveContainer: LinearLayout? = null
    private var centerPulse: View? = null
    private var btnStop: Button? = null // Đổi thành nút X (hủy)
    private var ivMicrophone: ImageView? = null
    
    // Status and message cards
    private var statusCard: android.view.View? = null
    private var tvStatusIcon: TextView? = null
    private var tvStatusText: TextView? = null
    private var confirmationCard: android.view.View? = null
    private var tvConfirmationText: TextView? = null
    private var successCard: android.view.View? = null
    private var tvSuccessText: TextView? = null
    private var errorCard: android.view.View? = null
    private var tvErrorText: TextView? = null

    // Animation Handler
    private val uiHandler = Handler(Looper.getMainLooper())
    private var animationRunnable: Runnable? = null
    private var startTime: Long = 0

    // Liquid wave bars
    private val waveBars = mutableListOf<View>()
    private val waveAnimations = mutableListOf<AnimatorSet>()

    // Voice responsiveness vars
    private var voiceIntensity = 0f
    private var smoothVoiceIntensity = 0f
    private var isLoudVoice = false
    private var isSoftVoice = false

    // Callbacks (Khớp với FloatingWindow)
    var onStopClick: (() -> Unit)? = null

    init {
        setupWindowManager()
        createPopupView()
    }

    private fun setupWindowManager() {
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    }

    private fun createPopupView() {
        val inflater = LayoutInflater.from(context)
        // Đảm bảo tên layout trong project của bạn là sound_wave_recording_popup
        popupView = inflater.inflate(R.layout.sound_wave_recording_popup, null)

        // Initialize UI components
        tvTimer = popupView?.findViewById(R.id.tv_timer)
        soundWaveContainer = popupView?.findViewById(R.id.sound_wave_container)
        centerPulse = popupView?.findViewById(R.id.center_pulse)
        ivMicrophone = popupView?.findViewById(R.id.iv_microphone)

        // Status and message cards
        statusCard = popupView?.findViewById(R.id.status_card)
        tvStatusIcon = popupView?.findViewById(R.id.tv_status_icon)
        tvStatusText = popupView?.findViewById(R.id.tv_status_text)
        confirmationCard = popupView?.findViewById(R.id.confirmation_card)
        tvConfirmationText = popupView?.findViewById(R.id.tv_confirmation_text)
        successCard = popupView?.findViewById(R.id.success_card)
        tvSuccessText = popupView?.findViewById(R.id.tv_success_text)
        errorCard = popupView?.findViewById(R.id.error_card)
        tvErrorText = popupView?.findViewById(R.id.tv_error_text)

        // Setup Cancel button (X)
        btnStop = popupView?.findViewById(R.id.btn_toggle_record)

        btnStop?.setOnClickListener {
            // Khi bấm nút X, hủy lệnh và tắt popup
            onStopClick?.invoke()
        }

        createLiquidWaveBars()
    }

    private fun createLiquidWaveBars() {
        soundWaveContainer?.let { container ->
            container.removeAllViews()
            waveBars.clear()
            waveAnimations.clear()

            for (i in 0 until 20) {
                val bar = View(context).apply {
                    layoutParams = LinearLayout.LayoutParams(
                        dpToPx(10),
                        dpToPx(16)
                    ).apply {
                        marginEnd = dpToPx(3)
                    }
                    // Đảm bảo drawable này tồn tại
                    setBackgroundResource(R.drawable.liquid_wave_bar)
                    alpha = 0.6f
                }
                container.addView(bar)
                waveBars.add(bar)
            }
        }
    }

    fun show() {
        if (isShowing) return

        try {
            if (popupView == null) createPopupView()

            val layoutParams = WindowManager.LayoutParams().apply {
                type = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                format = PixelFormat.TRANSLUCENT
                width = WindowManager.LayoutParams.WRAP_CONTENT
                height = WindowManager.LayoutParams.WRAP_CONTENT
                gravity = Gravity.CENTER
            }

            windowManager?.addView(popupView, layoutParams)
            isShowing = true
            startTime = System.currentTimeMillis()

            // Reset state
            smoothVoiceIntensity = 0f
            voiceIntensity = 0f
            updateStatus("")
            updateConfirmation("")
            updateSuccess("")
            updateError("")

            // Start animations
            startTimer()
            startLiquidWaveAnimation()
            startCenterPulseAnimation()
            startMicrophoneBreathing()

            Log.d("LiquidPopup", "Shown")
        } catch (e: Exception) {
            Log.e("LiquidPopup", "Error showing: ${e.message}")
        }
    }

    fun hide() {
        if (!isShowing) return

        try {
            stopAllAnimations()
            windowManager?.removeView(popupView)
            isShowing = false
            Log.d("LiquidPopup", "Hidden")
        } catch (e: Exception) {
            Log.e("LiquidPopup", "Error hiding: ${e.message}")
        }
    }

    /**
     * Hàm cập nhật trạng thái (giống SmoothStatusCard)
     */
    fun updateStatus(text: String) {
        uiHandler.post {
            if (text.isNotEmpty()) {
                tvStatusText?.text = text
                tvStatusIcon?.text = "🎤"
                statusCard?.visibility = android.view.View.VISIBLE
            } else {
                statusCard?.visibility = android.view.View.GONE
            }
        }
    }
    
    /**
     * Hàm cập nhật confirmation question (giống SmoothConfirmationCard)
     */
    fun updateConfirmation(question: String) {
        uiHandler.post {
            if (question.isNotEmpty()) {
                tvConfirmationText?.text = question
                confirmationCard?.visibility = android.view.View.VISIBLE
            } else {
                confirmationCard?.visibility = android.view.View.GONE
            }
        }
    }
    
    /**
     * Hàm cập nhật success message (giống SmoothSuccessCard)
     */
    fun updateSuccess(message: String) {
        uiHandler.post {
            if (message.isNotEmpty()) {
                tvSuccessText?.text = message
                successCard?.visibility = android.view.View.VISIBLE
            } else {
                successCard?.visibility = android.view.View.GONE
            }
        }
    }
    
    /**
     * Hàm cập nhật error message (giống SmoothErrorCard)
     */
    fun updateError(error: String) {
        uiHandler.post {
            if (error.isNotEmpty()) {
                tvErrorText?.text = error
                errorCard?.visibility = android.view.View.VISIBLE
            } else {
                errorCard?.visibility = android.view.View.GONE
            }
        }
    }

    /**
     * Hàm cập nhật Audio Level (Khớp tên với FloatingWindow gọi)
     */
    fun updateAudioLevel(level: Int) {
        // Đảm bảo chạy trên Main Thread để update Animation
        uiHandler.post {
            // Level 0-3 hoặc 0-10 -> convert sang 0.0 - 1.0
            val rawIntensity = (level.toFloat() / 3f).coerceIn(0f, 1f)

            // EMA filter
            smoothVoiceIntensity = 0.6f * smoothVoiceIntensity + 0.4f * rawIntensity
            voiceIntensity = smoothVoiceIntensity

            isLoudVoice = voiceIntensity >= 0.6f
            isSoftVoice = voiceIntensity <= 0.3f

            updateLiquidWaveBars()
            updateCenterPulse()
        }
    }

    private fun startTimer() {
        animationRunnable = object : Runnable {
            override fun run() {
                if (isShowing) {
                    val elapsed = System.currentTimeMillis() - startTime
                    val seconds = (elapsed / 1000).toInt()
                    val minutes = seconds / 60
                    val remainingSeconds = seconds % 60

                    tvTimer?.text = String.format("%02d:%02d", minutes, remainingSeconds)
                    uiHandler.postDelayed(this, 1000)
                }
            }
        }
        uiHandler.post(animationRunnable!!)
    }

    private fun startLiquidWaveAnimation() {
        waveBars.forEachIndexed { index, bar ->
            val animatorSet = AnimatorSet()

            val heightAnimator = ValueAnimator.ofFloat(0.3f, 1.0f).apply {
                duration = (600 + (index * 80)).toLong()
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()

                addUpdateListener { animator ->
                    if (!isShowing) return@addUpdateListener
                    val baseScale = animator.animatedValue as Float
                    // Sử dụng voiceIntensity được update từ hàm updateAudioLevel
                    val voiceScale = 1f + (voiceIntensity * 0.5f)
                    val liquidScale = baseScale * voiceScale

                    val newHeight = (dpToPx(16) * liquidScale).toInt()
                    bar.layoutParams.height = newHeight
                    bar.requestLayout()
                }
            }

            val alphaAnimator = ObjectAnimator.ofFloat(bar, "alpha", 0.3f, 1.0f).apply {
                duration = (500 + (index * 60)).toLong()
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }

            val rotationAnimator = ObjectAnimator.ofFloat(bar, "rotation", -5f, 5f).apply {
                duration = (1000 + (index * 120)).toLong()
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }

            val scaleXAnimator = ObjectAnimator.ofFloat(bar, "scaleX", 0.8f, 1.2f).apply {
                duration = (800 + (index * 100)).toLong()
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }

            animatorSet.playTogether(heightAnimator, alphaAnimator, rotationAnimator, scaleXAnimator)
            animatorSet.start()
            waveAnimations.add(animatorSet)
        }
    }

    private fun startCenterPulseAnimation() {
        centerPulse?.let { pulse ->
            val animatorSet = AnimatorSet()
            val scaleX = ObjectAnimator.ofFloat(pulse, "scaleX", 1.0f, 3.0f).apply {
                duration = 1000; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE
            }
            val scaleY = ObjectAnimator.ofFloat(pulse, "scaleY", 1.0f, 3.0f).apply {
                duration = 1000; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE
            }
            val alpha = ObjectAnimator.ofFloat(pulse, "alpha", 0.8f, 0.1f).apply {
                duration = 1000; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE
            }
            animatorSet.playTogether(scaleX, scaleY, alpha)
            animatorSet.start()
            waveAnimations.add(animatorSet)
        }
    }

    private fun startMicrophoneBreathing() {
        ivMicrophone?.let { mic ->
            val animatorSet = AnimatorSet()
            val scaleX = ObjectAnimator.ofFloat(mic, "scaleX", 1.0f, 1.2f).apply {
                duration = 800; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE
            }
            val scaleY = ObjectAnimator.ofFloat(mic, "scaleY", 1.0f, 1.2f).apply {
                duration = 800; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE
            }
            val alpha = ObjectAnimator.ofFloat(mic, "alpha", 0.8f, 1.0f).apply {
                duration = 800; repeatCount = ValueAnimator.INFINITE; repeatMode = ValueAnimator.REVERSE
            }
            animatorSet.playTogether(scaleX, scaleY, alpha)
            animatorSet.start()
            waveAnimations.add(animatorSet)
        }
    }

    private fun stopAllAnimations() {
        animationRunnable?.let { uiHandler.removeCallbacks(it) }
        waveAnimations.forEach { it.cancel() }
        waveAnimations.clear()
    }

    private fun updateLiquidWaveBars() {
        waveBars.forEachIndexed { index, bar ->
            val randomFactor = 0.8f + (Random.nextFloat() * 0.4f)

            if (isLoudVoice) {
                // Loud Logic
                val loudScale = 1.3f + (voiceIntensity * 0.4f)
                bar.scaleX = loudScale * randomFactor
                bar.alpha = (0.7f + voiceIntensity * 0.3f).coerceIn(0.5f, 1.0f)
            } else if (isSoftVoice) {
                // Soft Logic
                val softScale = 0.9f + (voiceIntensity * 0.3f)
                bar.scaleX = softScale * randomFactor
                bar.alpha = (0.5f + voiceIntensity * 0.2f).coerceIn(0.4f, 0.8f)
            } else {
                // Normal Logic
                val normalScale = 1f + (voiceIntensity * 0.2f)
                bar.scaleX = normalScale * randomFactor
                bar.alpha = (0.6f + voiceIntensity * 0.4f).coerceIn(0.4f, 1.0f)
            }
        }
    }

    private fun updateCenterPulse() {
        centerPulse?.let { pulse ->
            if (isLoudVoice) {
                pulse.scaleX = 2.5f + (voiceIntensity * 1.5f)
                pulse.scaleY = 2.5f + (voiceIntensity * 1.5f)
                pulse.alpha = 0.9f
            } else {
                pulse.scaleX = 2f + (voiceIntensity * 0.8f)
                pulse.scaleY = 2f + (voiceIntensity * 0.8f)
                pulse.alpha = 0.7f
            }
        }
    }


    private fun dpToPx(dp: Int): Int {
        return (dp * context.resources.displayMetrics.density).toInt()
    }

    fun isVisible(): Boolean = isShowing
}