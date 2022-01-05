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

static void UpdateFolders(CommandLine& args)
{
    args.face_cascade_name = Join(args.data_folder, "haarcascade_frontalface_alt.xml", '/');
    args.shape_predictor_path = Join(args.data_folder, "shape_predictor_68_face_landmarks.dat", '/');

    args.ALARM_LOC = Join(args.data_folder, "alarm.wav", '/');
    args.CALIBRATION_START_LOC = Join(args.data_folder, "calibration_start.wav", '/');
    args.CALIBRATION_END_LOC = Join(args.data_folder, "calibration_complete.wav", '/');
}

CommandLine::CommandLine()
{
    Init(0, nullptr);
}
CommandLine::CommandLine(int argc, const char** argv)
{
    Init(argc, argv);
}

void CommandLine::Init(int argc, const char** argv)
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

    UpdateFolders(*this);

    if (argc > 0 && argv != nullptr)
    {
        cli::Parser parser(argc, argv);
        parser.enable_help();
        parser.set_optional<std::string>("d", "data-folder", data_folder, "Folder to read data-files from");
        parser.set_optional<bool>("l", "--log", log_events_stdout, "Output detection messages to stdout");
        parser.set_optional<bool>("s", "--video", show_video, "Enable showing video");
        parser.set_optional<bool>("c", "--save-capture", capture_to_file, "Save captured video");
        parser.set_optional<bool>("e", "--show-eyes", show_eye_contour, "Show eye-contour");
        parser.set_optional<std::string>("a", "--alarm", ALARM_LOC, "Alarm wav file");
        parser.set_optional<bool>("f", "--show-face", show_face_detection, "Show wether or not face is detected");
        parser.set_optional<bool>("a", "--show-landmarks", show_all_factial_landmarks, "Shows all 68 facial landmarks on the face");

        parser.run_and_exit_if_error();
    
        UpdateFolders(*this);
    }
}

CommandLine::~CommandLine()
{}
