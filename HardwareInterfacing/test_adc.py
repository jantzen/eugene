#!/usr/bin/python3

# Raspberry Pi LPC1114 I/O Processor Expansion Board SPI Agent Firmware
# analog input test program

# Copyright (C)2013-2015, Philip Munts, President, Munts AM Corp.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# $Id$

# The following line was added by Jantzen
from __future__ import print_function

import sys
import time
import spiagent

print('\nRaspberry Pi LPC1114 I/O Processor Expansion Board Analog Input Test\n')

# Validate parameters

if len(sys.argv) == 1:
  server = 'localhost'
  transport = 'auto'
elif len(sys.argv) == 2:
  server = sys.argv[1]
  transport = 'auto'
elif len(sys.argv) == 3:
  server = sys.argv[1]
  transport = sys.argv[2]
else:
  print('Usage: ' + sys.argv[0] + ' [<hostname>] [auto|libspiagent|xml-rpc]')
  sys.exit(1)

# Open connection to the server

try:
  t = spiagent.Transport(server, transport)
except:
  print('ERROR: ' + str(sys.exc_info()[1]))
  sys.exit(1)

# Configure analog input pins

AnalogInputs = {channel: spiagent.ADC(t, spiagent.LPC1114_AD1 + channel - 1) for channel in range(spiagent.ANALOG_MIN_CHANNEL, spiagent.ANALOG_MAX_CHANNEL+1)}

print('Press CONTROL-C to quit\n')

try:
  while True:

# Sample analog inputs

    for channel in range(spiagent.ANALOG_MIN_CHANNEL, spiagent.ANALOG_MAX_CHANNEL+1):
      print('AD' + str(channel) + ': {:.2f} V   '.format(AnalogInputs[channel].voltage), end='')

    print('\r', end='')
    time.sleep(1)

# Handle CONTROL-C

except KeyboardInterrupt:

# Graceful shutdown

  print('')
  t.close()
  sys.exit(0)
