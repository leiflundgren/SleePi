#include "../include/CommandLine.h"
#include "../include/cmdparser.hpp"


static void UpdateFolders(CommandLine& args)
{
    args.face_cascade_name = args.data_folder + "haarcascade_frontalface_alt.xml";
    args.shape_predictor_path = args.data_folder + "shape_predictor_68_face_landmarks.dat";

    args.ALARM_LOC = args.data_folder + "alarm.wav";
    args.CALIBRATION_START_LOC = args.data_folder + "calibration_start.wav";
    args.CALIBRATION_END_LOC = args.data_folder + "calibration_complete.wav";
}

CommandLine::CommandLine()
{
    Init(0, nullptr);
}
CommandLine::CommandLine(int argc, char** argv)
{
    Init(argc, argv);
}

void CommandLine::Init(int argc, char** argv)
{
    data_folder = "../static/";
    show_video = true;
    capture_filename = "./capture.avi";
    capture_to_file = false;
    show_eye_contour = true;
    show_ear_score = true;

    UpdateFolders(*this);

    cli::Parser parser(argc, argv);
    parser.enable_help();
    parser.set_optional<std::string>("d", "data-folder", data_folder, "Folder to read data-files from");
    parser.set_optional<bool>("v", "--video", show_video, "Enable showing video");
    parser.set_optional<bool>("c", "--save-capture", capture_to_file, "Save captured video");
    parser.set_optional<bool>("e", "--show-eyes", show_eye_contour, "Show eye-contour");
    parser.set_optional<std::string>("a", "--alarm", ALARM_LOC, "Alarm wav file");

    parser.run_and_exit_if_error();


    UpdateFolders(*this);

}

CommandLine::~CommandLine()
{}
