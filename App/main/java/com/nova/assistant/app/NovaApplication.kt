package com.nova.assistant.app

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class NovaApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        // Initialization that doesn't require permissions can go here
    }
}
