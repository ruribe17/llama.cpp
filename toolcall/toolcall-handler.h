#pragma once

#include "toolcall-params.h"
#include <string>
#include <optional>
#include <memory>
#include <vector>
#include <condition_variable>
#include <mutex>

namespace toolcall
{
    struct result {
        std::string type;
        std::string data;
        std::string mime_type;
        std::optional<std::string> uri;
        bool error;
    };

    using result_set = std::vector<result>;

    class handler_impl;
    class handler {
    public:
        using ptr = std::shared_ptr<handler>;

        handler(std::unique_ptr<handler_impl> impl) : impl_(std::move(impl)) {}

        result_set call(const std::string & request);

        std::string tool_list();
        bool tool_list_dirty() const;

        const std::string & tool_choice() const;

        void initialize();

    private:
        std::unique_ptr<handler_impl> impl_;
    };

    std::shared_ptr<toolcall::handler> create_handler(const toolcall::params & params);

    class handler_impl {
    public:
        handler_impl(std::string tool_choice)
            : tool_choice_(std::move(tool_choice)), tool_list_dirty_(true) {}

        virtual ~handler_impl() = default;

        virtual std::string tool_list() = 0;

        virtual bool tool_list_dirty() const {
            return tool_list_dirty_;
        }

        virtual result_set call(const std::string & request) = 0;

        const std::string & tool_choice() const { return tool_choice_; }

        virtual void initialize() {}

    protected:
        std::string tool_choice_;
        bool tool_list_dirty_;
    };

    class loopback_impl : public handler_impl {
    public:
        loopback_impl(std::string tools, std::string tool_choice)
            : handler_impl(tool_choice), tools_(std::move(tools)) {}

        virtual std::string tool_list() override {
            tool_list_dirty_ = false;
            return tools_;
        }

        virtual result_set call(const std::string & request) override {
            return result_set {
                {"text", request, "text/plain", std::nullopt, false}
            };
        }

    private:
        std::string tools_;
    };

    class mcp_transport;
    class mcp_impl : public handler_impl {
    public:
        mcp_impl(std::string server_uri, std::string tool_choice);
        mcp_impl(std::vector<std::string> argv, std::string tool_choice);

        virtual std::string tool_list() override;
        virtual result_set call(const std::string & request) override;

        virtual void initialize() override;

    private:
        std::unique_ptr<mcp_transport> transport_;
        std::string tools_;
        std::mutex tools_mutex_;
        std::condition_variable tools_populating_;
        int next_id_;
    };
}
