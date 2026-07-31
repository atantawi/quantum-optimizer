"""qsim-service client: the HTTP/JSON boundary to the simulation engine.

qsim-service is GPL v2 because it links JMT in-process. qopt is Apache-2.0 and speaks
only HTTP/JSON to it, never importing or linking JMT code. That boundary is the
licensing firewall, so nothing in this subpackage may grow a runtime dependency.
"""
