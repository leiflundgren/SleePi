#pragma once

#include <string>

/// <summary>
/// Implemented through https://github.com/FlorianRappl/CmdParser, v1.1.0
/// 
/// </summary>
struct CommandLine
{
    CommandLine(int argc, char** argv);
    CommandLine();
    ~CommandLine();


    void Init(int argc, char** argv);

    std::string data_folder;
    bool show_video;
    bool capture_to_file;
    bool show_eye_contour;
    bool show_ear_score;
    std::string capture_filename;


    std::string face_cascade_name;
    std::string shape_predictor_path;

    std::string ALARM_LOC;
    std::string CALIBRATION_START_LOC;
    std::string CALIBRATION_END_LOC;

};