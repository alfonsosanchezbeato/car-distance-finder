#include <boost/preprocessor.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#define X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE(r, data, elem)    \
    case elem : return BOOST_PP_STRINGIZE(elem);

#define ENUM_WITH_STRINGS(name, enumerators)                                  \
    enum name {                                                               \
        BOOST_PP_SEQ_ENUM(enumerators)                                        \
    };                                                                        \
                                                                              \
    inline static const char* name##Str(name v)                               \
    {                                                                         \
        switch (v)                                                            \
        {                                                                     \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE,          \
                name,                                                         \
                enumerators                                                   \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }

// Logging macros
boost::log::trivial::severity_level severity_level_g;

void initLog(boost::log::trivial::severity_level lvl)
{
    severity_level_g = lvl;

    boost::log::core::get()->set_filter(boost::log::trivial::severity >= lvl);
}

#define LOG(x) BOOST_LOG_TRIVIAL(x)
#define TRACE_CODE(x, y) if (boost::log::trivial::x >= severity_level_g) { y }
