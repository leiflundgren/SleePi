#include "../include/CommandLine.h"
#include "../include/cmdparser.hpp"

static std::string Join(std::string s1, std::string s2, char separator)
{
    if (s1.empty())
        return s2;
    if (s2.empty())
        return s1;
    if (s1[s1.size() - 1] == separator || s2[0] == separator)
        return s1 + s2; // no need for separator
    return s1 + separator + s2;
}


CommandLine::CommandLine()
{
    data_folder = "../static/";
    show_video = true;
    capture_filename = "./capture.avi";
    capture_to_file = false;
    show_eye_contour = true;
    show_ear_score = true;
    show_face_detection = true;
    show_all_factial_landmarks = false;
    log_events_stdout = true;
    play_alarm = true;

    face_cascade_name = Join(data_folder, "haarcascade_frontalface_alt.xml", '/');
    shape_predictor_path = Join(data_folder, "shape_predictor_68_face_landmarks.dat", '/');

    ALARM_LOC = Join(data_folder, "alarm.wav", '/');
    CALIBRATION_START_LOC = Join(data_folder, "calibration_start.wav", '/');
    CALIBRATION_END_LOC = Join(data_folder, "calibration_complete.wav", '/');
}

void CommandLine::Init(int argc, const char** argv)
{
    if (argc > 0 && argv != nullptr)
    {
        cli::Parser parser(argc, argv);
        parser.enable_help();
        parser.set_optional<std::string>("d", "data-folder", data_folder, "Folder to read data-files from");
        parser.set_optional<bool>("l", "log", log_events_stdout, "Output detection messages to stdout");
        parser.set_optional<bool>("s", "video", show_video, "Enable showing video");
        parser.set_optional<bool>("c", "save-capture", capture_to_file, "Save captured video");
        parser.set_optional<std::string>("o", "caputure-file", "", "Alarm wav file");
        parser.set_optional<bool>("e", "show-eyes", show_eye_contour, "Show eye-contour");
        parser.set_optional<bool>("alarm", "alarm", play_alarm, "Alarm wav file");
        parser.set_optional<std::string>("a", "alarm-file", "", "Alarm wav file");
        parser.set_optional<bool>("f", "show-face", show_face_detection, "Show wether or not face is detected");
        parser.set_optional<bool>("landmarks", "show-landmarks", show_all_factial_landmarks, "Shows all 68 facial landmarks on the face");
        parser.set_optional<bool>("ear", "show-ear", show_ear_score, "Shows EAR score");

        parser.run_and_exit_if_error();

        log_events_stdout = parser.get<bool>("l");
        show_video = parser.get<bool>("s");
        show_eye_contour = parser.get<bool>("e");
        show_face_detection = parser.get<bool>("f");
        show_all_factial_landmarks = parser.get<bool>("landmarks");
        show_ear_score = parser.get<bool>("ear");


        if ( !parser.get<std::string>("d").empty() )
            data_folder = parser.get<std::string>("d");

        play_alarm = parser.get<bool>("alarm");
        if (!parser.get<std::string>("a").empty())
            ALARM_LOC = parser.get<std::string>("a");
        else 
            ALARM_LOC = Join(data_folder, "alarm.wav", '/');

        capture_to_file = parser.get<bool>("c");
        if (!parser.get<std::string>("o").empty())
            capture_filename = parser.get<std::string>("o");


        CALIBRATION_START_LOC = Join(data_folder, "calibration_start.wav", '/');
        CALIBRATION_END_LOC = Join(data_folder, "calibration_complete.wav", '/');
        
    }
}

CommandLine::~CommandLine()
{}
