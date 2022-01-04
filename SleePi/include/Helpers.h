#pragma once
#include <time.h>


struct Timestamp
{
    Timestamp()
        : timestamp(time(NULL))
    {}

    Timestamp(time_t timestamp)
        : timestamp(timestamp)
    {}


    time_t timestamp;
};

template <typename StreamClass>
StreamClass& operator<<(StreamClass& strm, const Timestamp& ts)
{
    char buff[20];
    strftime(buff, 20, "%H:%M:%S ", localtime(&ts.timestamp));
    strm << buff;
    return strm;
}




class Helpers
{
};

