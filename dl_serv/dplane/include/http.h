#include <iostream>
#include <map>
#include <string>
#include <cstdio>
#include <functional>
#include <assert.h>
#include <regex>

#define CPPHTTPLIB_REQUEST_URI_MAX_LENGTH 8192

enum class HttpVersion {
	v1_0=0,
	v1_1
};


namespace detail {
struct ci {
	bool operator()(const std::string &s1, const std::string &s2) const {
		return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(), [](char c1, char c2){ return ::tolower(c1) < ::tolower(2); });
	}
};

}

typedef std::multimap<std::string, std::string, detail::ci> Headers;

struct Request {
	std::string method;
  std::string path;
  std::string target;
	std::string model;
  std::string model_version;
	std::string operation;
  std::string version;
	Headers headers;
	std::string body;

	bool has_header(const char*key) const;
	void set_header(const char *key, const char *val);
};

struct Response {
	std::string version;
	int status;
	Headers headers;
	std::string body;

	bool has_header(const char* key) const;
	void set_header(const char* key, const char *val);
  
};

class Stream {
	public:
		virtual ~Stream() {}
		virtual int read(char *ptr, size_t size) = 0;
		virtual int write(const char *ptr, size_t size) = 0;
		virtual int write(const char *ptr) = 0;
		virtual std::string get_remote_addr() const = 0;
		template <typename... Args>
		void write_format(const char *fmt, const Args &... args);
};

class BufferStream : public Stream {
	public:
		BufferStream() {}
		virtual ~BufferStream() {}
		virtual int read(char *ptr, size_t size);
		virtual int write(const char *ptr, size_t size);
		virtual int write(const char *ptr);
		virtual std::string get_remote_addr() const;
		const std::string &get_buffer() const;
	private:
		std::string buffer;
};

inline int BufferStream::read(char *ptr, size_t size) {
	return static_cast<int>(buffer.copy(ptr, size));
}

inline int BufferStream::write(const char *ptr, size_t size) {
	buffer.append(ptr, size);
	return static_cast<int>(size);
}

inline int BufferStream::write(const char *ptr) {
	size_t size = strlen(ptr);
	buffer.append(ptr, size);
	return static_cast<int>(size);
}

inline std::string BufferStream::get_remote_addr() const { return ""; }
inline const std::string &BufferStream::get_buffer() const { return buffer; }

template<typename... Args>
inline void Stream::write_format(const char *fmt, const Args &... args) {
	const auto bufsiz = 2048;
	char buf[bufsiz];
	auto n = snprintf(buf, bufsiz-1, fmt, args...);
	if (n>0) {
		if (n >=bufsiz -1) {
			std::vector<char> glowable_buf(bufsiz);
			while (n >= static_cast<int>(glowable_buf.size() -1)) {
				glowable_buf.resize(glowable_buf.size() * 2);
				n = snprintf(&glowable_buf[0], glowable_buf.size()-1, fmt, args...);
			}
			write(&glowable_buf[0], n);
		} else {
			write(buf, n);
		}
	}
}


template<typename T> 
inline void write_headers(Stream &strm, const T &info) {
	for (const auto &x : info.headers) {
		strm.write_format("%s: %s\r\n", x.first.c_str(), x.second.c_str());
	}
	strm.write("\r\n");
}

inline void write_request(BufferStream &bstrm, Request &req) {

	bstrm.write_format("%s %s HTTP/1.1\r\n", req.method.c_str(), req.path.c_str());

	if (req.body.empty()) {
		if (req.method == "POST" || req.method == "PUT" || req.method == "PATCH") {
			req.set_header("Content-Length", "0");
		} 
	} else {
		if (!req.has_header("Content-Length")) {
			auto length = std::to_string(req.body.size());
			req.set_header("Content-Length", length.c_str());
		}
	}

	write_headers(bstrm, req);

	if (!req.body.empty()) {
		bstrm.write(req.body.c_str(), req.body.size());
	}
}

inline bool Request::has_header(const char *key) const {
	return headers.find(key) != headers.end();
}

inline void Request::set_header(const char *key, const char *val) {
	headers.emplace(key, val);
}


inline const char* status_message(int status) {
	switch(status) {
		case 200: return "OK";
		case 301: return "Moved Permanently";
		case 302: return "Found";
		case 303: return "See Other";
		case 304: return "Not Modified";
		case 400: return "Bad Request";
		case 403: return "Forbidden";
		case 404: return "Not Found";
		case 413: return "Payload Too Large";
		case 414: return "Reques-URI Too Long";
		case 415: return "Unsupported Media Type";
		default:
		case 500: return "Internal Server Error";
	}
}


inline bool Response::has_header(const char *key) const {
	return headers.find(key) != headers.end();
}

inline void Response::set_header(const char *key, const char *val) {
	headers.emplace(key, val);
}

inline void write_response(BufferStream &bstrm, Response &res) {
	
	bstrm.write_format("HTTP/1.1 %d %s\r\n", res.status, status_message(res.status));
	if (res.body.empty()) {
		if (!res.has_header("Content-Length")) {
			res.set_header("Content-Length", "0");
		}
	} else {
		auto length = std::to_string(res.body.size());
		res.set_header("Content-Length", length.c_str());
	}

	write_headers(bstrm, res);

	if (!res.body.empty()) {
		bstrm.write(res.body.c_str(), res.body.size());
	}
}


class stream_line_reader {
public:
  stream_line_reader(Stream &strm, char *fixed_buffer, size_t fixed_buffer_size)
      : strm_(strm), fixed_buffer_(fixed_buffer),
        fixed_buffer_size_(fixed_buffer_size) {}

  const char *ptr() const {
    if (glowable_buffer_.empty()) {
      return fixed_buffer_;
    } else {
      return glowable_buffer_.data();
    }
  }

  size_t size() const {
    if (glowable_buffer_.empty()) {
      return fixed_buffer_used_size_;
    } else {
      return glowable_buffer_.size();
    }
  }

  bool getline() {
    fixed_buffer_used_size_ = 0;
    glowable_buffer_.clear();

    for (size_t i = 0;; i++) {
      char byte;
      auto n = strm_.read(&byte, 1);

      if (n < 0) {
        return false;
      } else if (n == 0) {
        if (i == 0) {
          return false;
        } else {
          break;
        }
      }

      append(byte);

      if (byte == '\n') { break; }
    }

    return true;
  }

private:
  void append(char c) {
    if (fixed_buffer_used_size_ < fixed_buffer_size_ - 1) {
      fixed_buffer_[fixed_buffer_used_size_++] = c;
      fixed_buffer_[fixed_buffer_used_size_] = '\0';
    } else {
      if (glowable_buffer_.empty()) {
        assert(fixed_buffer_[fixed_buffer_used_size_] == '\0');
        glowable_buffer_.assign(fixed_buffer_, fixed_buffer_used_size_);
      }
      glowable_buffer_ += c;
    }
  }

  Stream &strm_;
  char *fixed_buffer_;
  const size_t fixed_buffer_size_;
  size_t fixed_buffer_used_size_;
  std::string glowable_buffer_;
};

#if 0
inline bool read_headers(Stream &strm, Headers &headers) {
  static std::regex re(R"((.+?):\s*(.+?)\s*\r\n)");

  const auto bufsiz = 2048;
  char buf[bufsiz];

  stream_line_reader reader(strm, buf, bufsiz);

  for (;;) {
    if (!reader.getline()) { return false; }
    if (!strcmp(reader.ptr(), "\r\n")) { break; }
    std::cmatch m;
    if (std::regex_match(reader.ptr(), m, re)) {
      auto key = std::string(m[1]);
      auto val = std::string(m[2]);
      headers.emplace(key, val);
    }
  }

  return true;
}
#endif

inline bool parse_request_line(const char *s, Request &req) {
  static std::regex re("(GET|HEAD|POST|PUT|PATCH|DELETE|OPTIONS) "
                       "(([^?]+)(?:\\?(.+?))?) (HTTP/1\\.[01])\r\n");

  std::cmatch m;
  if (std::regex_match(s, m, re)) {
    req.version = std::string(m[5]);
    req.method = std::string(m[1]);
    req.target = std::string(m[2]);
    req.path = m[3];
    return true;
  }

  return false;
}

inline bool read_headers(Stream &strm, Headers &headers) {
  static std::regex re(R"((.+?):\s*(.+?)\s*\r\n)");

  const auto bufsiz = 2048;
  char buf[bufsiz];

  stream_line_reader reader(strm, buf, bufsiz);

  for (;;) {
    if (!reader.getline()) { return false; }
    if (!strcmp(reader.ptr(), "\r\n")) { break; }
    std::cmatch m;
    if (std::regex_match(reader.ptr(), m, re)) {
      auto key = std::string(m[1]);
      auto val = std::string(m[2]);
      headers.emplace(key, val);
    }
  }

  return true;
}

inline bool
process_header(Stream &strm, Request &req) {
  const auto bufsiz = 2048;
  char buf[bufsiz];

  stream_line_reader reader(strm, buf, bufsiz);

  // Connection has been closed on client
  if (!reader.getline()) { return false; }

#if 0
  // Check if the request URI doesn't exceed the limit
  if (reader.size() > CPPHTTPLIB_REQUEST_URI_MAX_LENGTH) {
    res.status = 414;
  }
  #endif

  // Request line and headers
  if (!parse_request_line(reader.ptr(), req) ||
      read_headers(strm, req.headers)) {
  //  res.status = 400;
    return true;
  }
  return true;
}