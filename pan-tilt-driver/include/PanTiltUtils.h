
#ifndef PAN_TILT_UTILS_H
#define PAN_TILT_UTILS_H

#define DEBUG 1

#define SERIAL_BAUD_RATE  115200
#define SERIAL_TIMEOUT    200
#define SERIAL_TERMINATOR '!'
#define SERIAL_SEPARATOR '/'
#define VALID_MSG_LENGTH 11

#define WAIT_FOR_SERIAL() while(!Serial){}

#ifndef FORMAT
#define FORMAT(data) (String(data).c_str())
#endif

#define MASK_STR "%*"
#define MASK_LEN 2

String _REPLACE_FIRST_MASK_(String str, String arg);

void DEBUG_MSG (String msg, ...)
{
    String str, out;
    va_list arguments;                     
    
    if (DEBUG)
    {
        va_start (arguments, msg);
        out = msg;
        
        do
        {
            str = va_arg(arguments, char*);
            if (str == msg) continue;
            out = _REPLACE_FIRST_MASK_(out, str);
        } while (str != msg);

        va_end (arguments);

        Serial.println(String("DEBUG MESSAGE: " + String(out)));
        Serial.flush();
    }
}

String _REPLACE_FIRST_MASK_(String str, String arg)
{
    String _str = str.substring(str.indexOf(MASK_STR) + MASK_LEN, str.length());
    str.remove(str.indexOf(MASK_STR) + MASK_LEN, str.length() - (str.indexOf(MASK_STR) + MASK_LEN));
    str.replace(MASK_STR, arg);
    str.concat(_str);
    return str;
}

#endif
