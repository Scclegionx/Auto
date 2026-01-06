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
import android.view.ViewGroup
import android.view.WindowManager
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.auto_fe.auto_fe.R
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random

/**
 * Sound Wave Recording Popup với animation mềm mại
 * Hiển thị sound wave bars và center pulse animation
 */
class SoundWaveRecordingPopup(private val context: Context) {
    
    private var windowManager: WindowManager? = null
    private var popupView: View? = null
    private var isShowing = false
    
    // UI components
    private var tvTimer: TextView? = null
    private var tvTranscript: TextView? = null
    private var soundWaveContainer: LinearLayout? = null
    private var centerPulse: View? = null
    private var btnCancel: Button? = null
    private var btnStop: Button? = null
    
    // Animation
    private val animationHandler = Handler(Looper.getMainLooper())
    private var animationRunnable: Runnable? = null
    private var startTime: Long = 0
    private var currentTranscript = ""
    
    // Sound wave bars
    private val waveBars = mutableListOf<View>()
    private val waveAnimations = mutableListOf<AnimatorSet>()
    
    // Callbacks
    var onCancelClick: (() -> Unit)? = null
    var onStopClick: (() -> Unit)? = null
    var onAudioLevelChanged: ((Int) -> Unit)? = null
    
    init {
        setupWindowManager()
        createPopupView()
    }
    
    private fun setupWindowManager() {
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    }
    
    private fun createPopupView() {
        val inflater = LayoutInflater.from(context)
        popupView = inflater.inflate(R.layout.sound_wave_recording_popup, null)
        
        // Initialize UI components
        tvTimer = popupView?.findViewById(R.id.tv_timer)
        tvTranscript = popupView?.findViewById(R.id.tv_transcript)
        soundWaveContainer = popupView?.findViewById(R.id.sound_wave_container)
        centerPulse = popupView?.findViewById(R.id.center_pulse)
        btnCancel = popupView?.findViewById(R.id.btn_cancel)
        btnStop = popupView?.findViewById(R.id.btn_stop)
        
        // Setup click listeners
        btnCancel?.setOnClickListener {
            onCancelClick?.invoke()
            hide()
        }
        
        btnStop?.setOnClickListener {
            onStopClick?.invoke()
            hide()
        }
        
        // Create sound wave bars
        createSoundWaveBars()
    }
    
    private fun createSoundWaveBars() {
        soundWaveContainer?.let { container ->
            // Clear existing bars
            container.removeAllViews()
            waveBars.clear()
            waveAnimations.clear()
            
            // Create 20 wave bars
            for (i in 0 until 20) {
                val bar = View(context).apply {
                    layoutParams = LinearLayout.LayoutParams(
                        dpToPx(8), // 8dp width
                        dpToPx(12) // 12dp initial height
                    ).apply {
                        marginEnd = dpToPx(2) // 2dp margin
                    }
                    setBackgroundResource(R.drawable.sound_wave_bar)
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
            if (popupView == null) {
                Log.e("SoundWaveRecordingPopup", "Popup view is null, recreating...")
                createPopupView()
            }
            
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
            
            // Start animations with delay
            Handler(Looper.getMainLooper()).postDelayed({
                if (isShowing) {
                    startTimer()
                    startSoundWaveAnimation()
                    startCenterPulseAnimation()
                }
            }, 200)
            
            Log.d("SoundWaveRecordingPopup", "Sound wave popup shown")
        } catch (e: Exception) {
            Log.e("SoundWaveRecordingPopup", "Error showing popup: ${e.message}")
            isShowing = false
        }
    }
    
    fun hide() {
        if (!isShowing) return
        
        try {
            stopAllAnimations()
            windowManager?.removeView(popupView)
            isShowing = false
            Log.d("SoundWaveRecordingPopup", "Sound wave popup hidden")
        } catch (e: Exception) {
            Log.e("SoundWaveRecordingPopup", "Error hiding popup: ${e.message}")
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
                    animationHandler.postDelayed(this, 1000)
                }
            }
        }
        animationHandler.post(animationRunnable!!)
    }
    
    private fun startSoundWaveAnimation() {
        waveBars.forEachIndexed { index, bar ->
            val animatorSet = AnimatorSet()
            
            // Height animation
            val heightAnimator = ValueAnimator.ofFloat(0.3f, 1.0f).apply {
                duration = (400 + (index * 50)).toLong() // Staggered timing
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
                
                addUpdateListener { animator ->
                    val scale = animator.animatedValue as Float
                    val newHeight = (dpToPx(12) * scale).toInt()
                    bar.layoutParams.height = newHeight
                    bar.requestLayout()
                }
            }
            
            // Alpha animation
            val alphaAnimator = ObjectAnimator.ofFloat(bar, "alpha", 0.4f, 1.0f).apply {
                duration = (300 + (index * 30)).toLong()
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            // Rotation animation for some bars
            val rotationAnimator = if (index % 3 == 0) {
                ObjectAnimator.ofFloat(bar, "rotation", -2f, 2f).apply {
                    duration = (800 + (index * 100)).toLong()
                    repeatCount = ValueAnimator.INFINITE
                    repeatMode = ValueAnimator.REVERSE
                    interpolator = AccelerateDecelerateInterpolator()
                }
            } else null
            
            animatorSet.playTogether(heightAnimator, alphaAnimator)
            rotationAnimator?.let { animatorSet.play(it) }
            
            animatorSet.start()
            waveAnimations.add(animatorSet)
        }
    }
    
    private fun startCenterPulseAnimation() {
        centerPulse?.let { pulse ->
            val animatorSet = AnimatorSet()
            
            val scaleX = ObjectAnimator.ofFloat(pulse, "scaleX", 1.0f, 2.5f).apply {
                duration = 800
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            val scaleY = ObjectAnimator.ofFloat(pulse, "scaleY", 1.0f, 2.5f).apply {
                duration = 800
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            val alpha = ObjectAnimator.ofFloat(pulse, "alpha", 0.8f, 0.2f).apply {
                duration = 800
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            animatorSet.playTogether(scaleX, scaleY, alpha)
            animatorSet.start()
            waveAnimations.add(animatorSet)
        }
    }
    
    private fun stopAllAnimations() {
        animationRunnable?.let { animationHandler.removeCallbacks(it) }
        waveAnimations.forEach { it.cancel() }
        waveAnimations.clear()
    }
    
    fun updateTranscript(transcript: String) {
        currentTranscript = transcript
        tvTranscript?.text = if (transcript.isNotEmpty()) transcript else "Đang lắng nghe..."
    }
    
    fun updateAudioLevel(level: Int) {
        // Update wave bars based on audio level
        val intensity = (level / 100f).coerceIn(0.3f, 1.0f)
        
        waveBars.forEachIndexed { index, bar ->
            val randomFactor = 0.7f + (Random.nextFloat() * 0.6f)
            val newAlpha = (intensity * randomFactor).coerceIn(0.4f, 1.0f)
            bar.alpha = newAlpha
        }
    }
    
    private fun dpToPx(dp: Int): Int {
        return (dp * context.resources.displayMetrics.density).toInt()
    }
    
    fun isVisible(): Boolean = isShowing
}
