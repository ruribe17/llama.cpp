#pragma once

#include "mcp_messages.h"
#include <string>
#include <tuple>
#include <map>

namespace toolcall
{
    template <typename T>
    using callback = std::function<void(const T&)>;

    template <typename... MessageTypes>
    class mcp_message_observer {
    public:
        template <typename T>
        void subscribe(std::string key, callback<T> callback) {
            auto& map =
                std::get<std::map<std::string, toolcall::callback<T>>>(
                    subscribers_);

            map.insert({key, callback});
        }

        template <typename T>
        void unsubscribe(std::string key) {
            auto& map =
                std::get<std::map<std::string, toolcall::callback<T>>>(
                    subscribers_);

            map.erase(key);
        }

        template <typename T>
        void notify(const T & message) const {
            const auto& map =
                std::get<std::map<std::string, toolcall::callback<T>>>(
                    subscribers_);

            for (const auto & pair : map) {
                pair.second(message);
            }
        }

        template <typename T>
        void notify_if(const mcp::message_variant & message) {
            if (std::holds_alternative<T>(message)) {
                notify(std::get<T>(message));
            }
        }

    private:
        std::tuple<std::map<std::string, toolcall::callback<MessageTypes>>...> subscribers_;
    };

    class mcp_transport : public mcp_message_observer<mcp::initialize_request,
                                                      mcp::initialize_response,
                                                      mcp::initialized_notification,
                                                      mcp::tools_list_request,
                                                      mcp::tools_list_response,
                                                      mcp::tools_list_changed_notification> {
    public:
        virtual ~mcp_transport() = default;

        template <typename T>
        bool send(const T & message) {
            return send(std::string(message.toJson()));
        }

        virtual void start() = 0;
        virtual void stop() = 0;
        virtual bool send(const std::string & request_json) = 0;
    };
}
