#pragma once

#include <string>

/// <summary>
/// Implemented through https://github.com/FlorianRappl/CmdParser, v1.1.0
/// 
/// </summary>
struct CommandLine
{
    CommandLine(int argc, const char** argv);
    CommandLine();
    ~CommandLine();


    void Init(int argc, const char** argv);

    std::string data_folder;
    bool show_video;
    bool capture_to_file;
    bool show_eye_contour;
    bool show_ear_score;
    bool show_face_detection;
    bool show_all_factial_landmarks;
    bool log_events_stdout;
    std::string capture_filename;


    std::string face_cascade_name;
    std::string shape_predictor_path;

    std::string ALARM_LOC;
    std::string CALIBRATION_START_LOC;
    std::string CALIBRATION_END_LOC;

};