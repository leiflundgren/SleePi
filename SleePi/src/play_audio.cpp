#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <stdio.h>
#include <iostream>

#include "config.h"

#include "Helpers.h"

#include "play_audio.h"

ma_decoder decoder;
ma_device device;
ma_device_config config;
ma_uint32 g_decoderCount;
ma_decoder *g_pDecoders;
ma_event *g_stopEvent;

// Keeps track of which sample has finished playing
int sampleCounter = 0;
// True if currently playing sample has and end (does not loop)
bool isPlayingNonloopSample = false;
// Indicates whether a sample should be looped
bool loopSample = false;
// Indicates whether the alarm sample has been initialised
bool alarmReady = false;
// Indicates whether the alarm is currently on
bool alarmON = false;

// Initialises the alarm sample
int PlayAudio::init_alarm()
{
    // If this is true, it means that the vocal indication of the system's status has not yet finished playing and we should wait.
    if (isPlayingNonloopSample)
    {
        printf("waiting in alarm_init\n");
        ma_event_wait(&g_stopEvent[1]);
    }
    printf("done waiting in alarm_init\n");
    sampleCounter++;

    // Uninitialise the previous audio sample
    ma_decoder_uninit(&g_pDecoders[1]);
    ma_device_uninit(&device);

    loopSample = true;
    // Set data
    config.pUserData = &g_pDecoders[2];
    // Init device
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS)
    {
        printf("Failed to open playback device.\n");
        ma_decoder_uninit(&decoder);
        return -3;
    }
    alarmReady = true;
    return 0;
}
// Starts playing the alarm sammple
int PlayAudio::start_alarm()
{
    printf("Start alarm called \n");
    alarmON = true;
    // Start playing
    if (ma_device_start(&device) != MA_SUCCESS)
    {
        printf("Failed to start playback device.\n");
        ma_device_uninit(&device);
        ma_decoder_uninit(&decoder);
        return -4;
    }
    return 0;
}
// Plays the audio sample indicating the start of calibration
int PlayAudio::play_calibration_start()
{
    printf("caliration start called\n");
    isPlayingNonloopSample = true;
    // This sample is played first, therefore we do not need to check status
    // Set data
    config.pUserData = &g_pDecoders[0];
    // Init device
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS)
    {
        printf("Failed to open playback device.\n");
        ma_decoder_uninit(&decoder);
        return -3;
    }
    // Start playing
    if (ma_device_start(&device) != MA_SUCCESS)
    {
        printf("Failed to start playback device.\n");
        ma_device_uninit(&device);
        ma_decoder_uninit(&decoder);
        return -4;
    }
    return 0;
}
// Plays the audio sample indicating the end of calibration
int PlayAudio::play_calibartion_completed()
{
    // If this is true then the previous audio sample has not finished, therefore it must wait
    if (isPlayingNonloopSample)
    {
        printf("waiting in play_calibartion_completed\n");
        ma_event_wait(&g_stopEvent[0]);
    }
    printf("done waiting in play_calibartion_completed \n");
    // Increment the counter to keep track which sample finished playing
    sampleCounter++;
    // Uninitialise the previous audio sample
    ma_decoder_uninit(&g_pDecoders[0]);
    ma_device_uninit(&device);

    // Set data
    config.pUserData = &g_pDecoders[1];
    // Init device
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS)
    {
        printf("Failed to open playback device.\n");
        ma_decoder_uninit(&decoder);
        return -3;
    }
    // Start playing
    if (ma_device_start(&device) != MA_SUCCESS)
    {
        printf("Failed to start playback device.\n");
        ma_device_uninit(&device);
        ma_decoder_uninit(&decoder);
        return -4;
    }
    return 0;
}
// Stops currently playing audio, should only be used to stop the looping alarm sample.
int PlayAudio::stop_playing()
{
    printf("Stopping playback \n");
    // We only want this function to execute if currently playing sound is the alarm.
    if (isPlayingNonloopSample)
    {
        return 1;
    }
    // Stop playing
    ma_device_stop(&device);
    alarmON = false;
    // No need to uninit device since we will likely use it again.
    return 0;
}
// Data callback function for playing audio
void data_callback(ma_device *pDevice, void *pOutput, const void *pInput, ma_uint32 framesToRead)
{
    MA_ASSERT(pDevice->playback.format == SAMPLE_FORMAT);
    // Set decoder to use our loaded sample
    ma_decoder *pDecoder = (ma_decoder *)pDevice->pUserData;
    if (pDecoder == NULL)
    {
        return;
    }
    // Data is read from the decoder as PCM frames. This will return the number of PCM frames actually read.
    ma_uint64 framesRead = ma_decoder_read_pcm_frames(pDecoder, pOutput, framesToRead);
    // If the return value is less than the requested number of PCM frames it means that end of the sample is reached
    if (framesRead < framesToRead)
    {
        // If true, then loop back to the start
        if (loopSample)
        {
            ma_decoder_seek_to_pcm_frame(pDecoder, 0);
        }
        // If not, then send a signal that the sample has finished playing
        else
        {
            ma_event_signal(&g_stopEvent[sampleCounter]);
            // Only set to false after all vocal instructions have finished.
            if (sampleCounter >= 1)
            {
                isPlayingNonloopSample = false;
            }
        }
    }

    (void)pInput;
}
// Loads the audio samples and initialises audio device
int PlayAudio::init_playback(const std::string& calibrate_start_file, const std::string& calibrate_finished_file, const std::string& alarm_file)
{
    ma_result result;

    // Equal to number of different files that will be played
    g_decoderCount = 3;
    g_pDecoders = (ma_decoder *)malloc(sizeof(*g_pDecoders) * g_decoderCount);
    g_stopEvent = (ma_event *)malloc(sizeof(*g_stopEvent) * g_decoderCount);

    // set up the same config for all decoders
    ma_decoder_config decoderConfig = ma_decoder_config_init(SAMPLE_FORMAT, CHANNEL_COUNT, SAMPLE_RATE);

    // Load alarm sound
    result = ma_decoder_init_file_wav(calibrate_start_file.c_str(), &decoderConfig, &g_pDecoders[0]);
    if (result != MA_SUCCESS)
    {
        std::cerr << "Failed to load file from " << calibrate_start_file<< "\n";
        return -2;
    }
    // Load calibration start sound
    result = ma_decoder_init_file_wav(calibrate_finished_file.c_str(), &decoderConfig, &g_pDecoders[1]);
    if (result != MA_SUCCESS)
    {
        std::cerr << "Failed to load file from " << calibrate_finished_file<< "\n";
        return -2;
    }

    // Load calibration completed sound
    if (!alarm_file.empty())
    {
        result = ma_decoder_init_file_wav(alarm_file.c_str(), &decoderConfig, &g_pDecoders[2]);
        if (result != MA_SUCCESS)
        {
            std::cerr << "Failed to load file from " << alarm_file << "\n";
            return -2;
        }
    }

    // Configure the device using speified parameters
    config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = SAMPLE_FORMAT;   // Set to ma_format_unknown to use the device's native format.
    config.playback.channels = CHANNEL_COUNT; // Set to 0 to use the device's native channel count.
    config.sampleRate = SAMPLE_RATE;          // Set to 0 to use the device's native sample rate.
    config.dataCallback = data_callback;      // This function will be called when miniaudio needs more data.

    // Initilise events that signal that the audio sample has finished. This is not done for the alarm sample because the alarm is looped
    ma_event_init(&g_stopEvent[0]);
    ma_event_init(&g_stopEvent[1]);

    return 0;
}