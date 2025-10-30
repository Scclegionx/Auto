package com.auto_fe.auto_fe.ui

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
import android.view.animation.OvershootInterpolator
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.auto_fe.auto_fe.R
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.cos
import kotlin.random.Random

/**
 * Liquid Wave Recording Popup với Voice-Responsive Enhancements
 * Hiển thị liquid wave bars và center pulse animation phản ứng với giọng nói
 */
class LiquidWaveRecordingPopup(private val context: Context) {
    
    private var windowManager: WindowManager? = null
    private var popupView: View? = null
    private var isShowing = false
    
    // UI components
    private var tvTimer: TextView? = null
    private var tvTranscript: TextView? = null
    private var soundWaveContainer: LinearLayout? = null
    private var centerPulse: View? = null
    private var btnToggleRecord: Button? = null
    private var ivMicrophone: ImageView? = null
    
    // Recording state
    private var isRecording = false
    
    // Animation
    private val animationHandler = Handler(Looper.getMainLooper())
    private var animationRunnable: Runnable? = null
    private var startTime: Long = 0
    private var currentTranscript = ""
    
    // Liquid wave bars
    private val waveBars = mutableListOf<View>()
    private val waveAnimations = mutableListOf<AnimatorSet>()
    
    // Voice responsiveness
    private var currentVoiceLevel = 0
    private var voiceIntensity = 0f
    private var smoothVoiceIntensity = 0f  // EMA filtered intensity
    private var isLoudVoice = false
    private var isSoftVoice = false
    
    // Liquid motion parameters
    private var liquidPhase = 0f
    private var liquidSpeed = 0.02f
    private var liquidAmplitude = 1f
    
    // Callbacks
    var onStartClick: (() -> Unit)? = null
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
        btnToggleRecord = popupView?.findViewById(R.id.btn_toggle_record)
        ivMicrophone = popupView?.findViewById(R.id.iv_microphone)
        
        // Setup toggle button click listener
        btnToggleRecord?.setOnClickListener {
            if (isRecording) {
                // Stop recording
                isRecording = false
                btnToggleRecord?.text = "REC"
                btnToggleRecord?.setBackgroundResource(R.drawable.red_circle_button)
                onStopClick?.invoke()
                hide()
            } else {
                // Start recording
                isRecording = true
                btnToggleRecord?.text = "STOP"
                btnToggleRecord?.setBackgroundResource(R.drawable.red_circle_button_recording)
                onStartClick?.invoke()
            }
        }
        
        // Create liquid wave bars
        createLiquidWaveBars()
    }
    
    private fun createLiquidWaveBars() {
        soundWaveContainer?.let { container ->
            // Clear existing bars
            container.removeAllViews()
            waveBars.clear()
            waveAnimations.clear()
            
            // Create 20 liquid wave bars
            for (i in 0 until 20) {
                val bar = View(context).apply {
                    layoutParams = LinearLayout.LayoutParams(
                        dpToPx(10), // 10dp width for liquid look
                        dpToPx(16) // 16dp initial height
                    ).apply {
                        marginEnd = dpToPx(3) // 3dp margin
                    }
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
            if (popupView == null) {
                Log.e("LiquidWaveRecordingPopup", "Popup view is null, recreating...")
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
            
            // Start animations immediately
            startTimer()
            startLiquidWaveAnimation()
            startCenterPulseAnimation()
            startMicrophoneBreathing()
            
            Log.d("LiquidWaveRecordingPopup", "Liquid wave popup shown")
        } catch (e: Exception) {
            Log.e("LiquidWaveRecordingPopup", "Error showing popup: ${e.message}")
            isShowing = false
        }
    }
    
    fun hide() {
        if (!isShowing) return
        
        try {
            stopAllAnimations()
            
            // Reset voice state
            currentVoiceLevel = 0
            voiceIntensity = 0f
            smoothVoiceIntensity = 0f
            isLoudVoice = false
            isSoftVoice = false
            
            windowManager?.removeView(popupView)
            isShowing = false
            Log.d("LiquidWaveRecordingPopup", "Liquid wave popup hidden")
        } catch (e: Exception) {
            Log.e("LiquidWaveRecordingPopup", "Error hiding popup: ${e.message}")
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
    
    private fun startLiquidWaveAnimation() {
        waveBars.forEachIndexed { index, bar ->
            val animatorSet = AnimatorSet()
            
            // Liquid height animation with voice responsiveness
            val heightAnimator = ValueAnimator.ofFloat(0.3f, 1.0f).apply {
                duration = (600 + (index * 80)).toLong() // Slower, more liquid-like
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
                
                addUpdateListener { animator ->
                    val baseScale = animator.animatedValue as Float
                    val voiceScale = 1f + (voiceIntensity * 0.5f) // Voice responsiveness
                    val liquidScale = baseScale * voiceScale
                    
                    val newHeight = (dpToPx(16) * liquidScale).toInt()
                    bar.layoutParams.height = newHeight
                    bar.requestLayout()
                }
            }
            
            // Liquid alpha animation
            val alphaAnimator = ObjectAnimator.ofFloat(bar, "alpha", 0.3f, 1.0f).apply {
                duration = (500 + (index * 60)).toLong()
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            // Liquid rotation animation for swirl effect
            val rotationAnimator = ObjectAnimator.ofFloat(bar, "rotation", -5f, 5f).apply {
                duration = (1000 + (index * 120)).toLong()
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            // Liquid scale animation for "breathing" effect
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
                duration = 1000
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            val scaleY = ObjectAnimator.ofFloat(pulse, "scaleY", 1.0f, 3.0f).apply {
                duration = 1000
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            val alpha = ObjectAnimator.ofFloat(pulse, "alpha", 0.8f, 0.1f).apply {
                duration = 1000
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
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
                duration = 800
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            val scaleY = ObjectAnimator.ofFloat(mic, "scaleY", 1.0f, 1.2f).apply {
                duration = 800
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.REVERSE
                interpolator = AccelerateDecelerateInterpolator()
            }
            
            val alpha = ObjectAnimator.ofFloat(mic, "alpha", 0.8f, 1.0f).apply {
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
        Log.d("LiquidWaveRecordingPopup", "updateAudioLevel called with level: $level")
        currentVoiceLevel = level
        
        // Level từ VoiceManager là 0-3, chuyển thành 0-1
        val rawIntensity = (level / 3f).coerceIn(0f, 1f)
        
        // EMA filter để smooth animation - giống VoiceScreen
        smoothVoiceIntensity = 0.6f * smoothVoiceIntensity + 0.4f * rawIntensity
        voiceIntensity = smoothVoiceIntensity
        
        // Determine voice characteristics dựa trên smoothed intensity
        isLoudVoice = voiceIntensity >= 0.6f  // 60% intensity trở lên là loud
        isSoftVoice = voiceIntensity <= 0.3f  // 30% intensity trở xuống là soft
        
        Log.d("LiquidWaveRecordingPopup", "Voice characteristics - isLoud: $isLoudVoice, isSoft: $isSoftVoice, rawIntensity: $rawIntensity, smoothIntensity: $voiceIntensity")
        
        // Update liquid wave bars based on voice level
        updateLiquidWaveBars()
        
        // Update center pulse based on voice
        updateCenterPulse()
        
        // Update microphone based on voice
        updateMicrophone()
    }
    
    private fun updateLiquidWaveBars() {
        waveBars.forEachIndexed { index, bar ->
            // Giảm random factor để animation mượt hơn
            val randomFactor = 0.8f + (Random.nextFloat() * 0.4f)
            val voiceScale = 1f + (voiceIntensity * 0.6f) // Giảm voice response để mượt hơn
            
            // Loud voice: stronger waves, brighter peaks
            if (isLoudVoice) {
                val loudScale = 1.3f + (voiceIntensity * 0.4f)
                bar.scaleX = loudScale * randomFactor
                bar.alpha = (0.7f + voiceIntensity * 0.3f).coerceIn(0.5f, 1.0f)
                
                // Giảm splash effect để mượt hơn
                bar.rotation = (Random.nextFloat() - 0.5f) * 5f
            }
            // Soft voice: gentle oscillations, soft light
            else if (isSoftVoice) {
                val softScale = 0.9f + (voiceIntensity * 0.3f)
                bar.scaleX = softScale * randomFactor
                bar.alpha = (0.5f + voiceIntensity * 0.2f).coerceIn(0.4f, 0.8f)
                
                // Gentle rotation for soft voice
                bar.rotation = (Random.nextFloat() - 0.5f) * 2f
            }
            // Normal voice: balanced response
            else {
                val normalScale = 1f + (voiceIntensity * 0.2f)
                bar.scaleX = normalScale * randomFactor
                bar.alpha = (0.6f + voiceIntensity * 0.4f).coerceIn(0.4f, 1.0f)
                
                // Subtle rotation
                bar.rotation = (Random.nextFloat() - 0.5f) * 5f
            }
        }
    }
    
    private fun updateCenterPulse() {
        centerPulse?.let { pulse ->
            if (isLoudVoice) {
                // Strong pulse for loud voice
                pulse.scaleX = 2.5f + (voiceIntensity * 1.5f)
                pulse.scaleY = 2.5f + (voiceIntensity * 1.5f)
                pulse.alpha = 0.9f
            } else if (isSoftVoice) {
                // Gentle pulse for soft voice
                pulse.scaleX = 1.5f + (voiceIntensity * 0.5f)
                pulse.scaleY = 1.5f + (voiceIntensity * 0.5f)
                pulse.alpha = 0.6f
            } else {
                // Normal pulse
                pulse.scaleX = 2f + (voiceIntensity * 0.8f)
                pulse.scaleY = 2f + (voiceIntensity * 0.8f)
                pulse.alpha = 0.7f
            }
        }
    }
    
    private fun updateMicrophone() {
        ivMicrophone?.let { mic ->
            if (isLoudVoice) {
                // Strong breathing for loud voice
                mic.scaleX = 1.3f + (voiceIntensity * 0.3f)
                mic.scaleY = 1.3f + (voiceIntensity * 0.3f)
                mic.alpha = 1.0f
            } else if (isSoftVoice) {
                // Gentle breathing for soft voice
                mic.scaleX = 1.1f + (voiceIntensity * 0.2f)
                mic.scaleY = 1.1f + (voiceIntensity * 0.2f)
                mic.alpha = 0.8f
            } else {
                // Normal breathing
                mic.scaleX = 1.2f + (voiceIntensity * 0.2f)
                mic.scaleY = 1.2f + (voiceIntensity * 0.2f)
                mic.alpha = 0.9f
            }
        }
    }
    
    private fun dpToPx(dp: Int): Int {
        return (dp * context.resources.displayMetrics.density).toInt()
    }
    
    fun isVisible(): Boolean = isShowing
}
