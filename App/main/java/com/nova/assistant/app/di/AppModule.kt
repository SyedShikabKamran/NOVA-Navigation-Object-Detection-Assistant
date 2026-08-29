package com.nova.assistant.app.di

import android.app.Application
import android.content.Context
import androidx.room.Room
import com.nova.assistant.data.local.NovaDatabase
import com.nova.assistant.data.local.RoomSnapshotDao
import com.nova.assistant.data.local.SettingsDao
import com.nova.assistant.server.NovaInferenceClient
import com.nova.assistant.util.FileLogger
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): NovaDatabase {
        return Room.databaseBuilder(
            context,
            NovaDatabase::class.java,
            NovaDatabase.DATABASE_NAME
        )
            .addMigrations(
                NovaDatabase.MIGRATION_2_3, NovaDatabase.MIGRATION_3_4,
                NovaDatabase.MIGRATION_4_5, NovaDatabase.MIGRATION_5_6,
            )
            .fallbackToDestructiveMigration()  // safety net for unexpected version gaps
            .build()
    }

    @Provides
    fun provideRoomSnapshotDao(db: NovaDatabase): RoomSnapshotDao = db.roomSnapshotDao()

    @Provides
    fun provideSettingsDao(db: NovaDatabase): SettingsDao = db.settingsDao()
}

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideAppContext(application: Application): Context = application.applicationContext

    @Provides
    @Singleton
    fun provideNovaInferenceClient(fileLogger: FileLogger): NovaInferenceClient =
        NovaInferenceClient(fileLogger)
}
