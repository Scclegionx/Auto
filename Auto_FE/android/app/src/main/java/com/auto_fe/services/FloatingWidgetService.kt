package com.auto_fe.services

import android.app.Service
import android.content.Intent
import android.os.IBinder
import com.auto_fe.widgets.FloatingWidget

class FloatingWidgetService : Service() {
    private lateinit var floatingWidget: FloatingWidget

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        floatingWidget = FloatingWidget(this)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        floatingWidget.destroy()
    }
} 