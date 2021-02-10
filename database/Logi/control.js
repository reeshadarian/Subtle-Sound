var patientName = 'Logi';
 var heartStats = 'HNR: 2.9336186939529116<br>Local Jitter: 0.1035875694189892<br>Local Absolute Jitter: 0.0008161858117740211<br>Local Shimmer: 0.15076671359081828<br>Local Shimmer dB: 1.8581199831069883<br>';
var breathStats = 'HNR: 2.655076075329061<br>Local Jitter: 0.0864072161682583<br>Local Absolute Jitter: 0.0006795173755246015<br>Local Shimmer: 0.24038081247244492<br>Local Shimmer dB: 2.089652281911529<br>';
var speechStats = 'Transcript: Laramie is an American television program recorded from the 1930s to 1940s this about tonight Sherman the owner of Sherman Ranch along with his younger brother Andy played by Robert Crawford and Dark Project l q Jones Ford<br>HNR: 6.498509469562061<br>Local Jitter: 0.03544919454728084<br>Local Absolute Jitter: 0.00017722429465072564<br>Local Shimmer: 0.197049263849013<br>Local Shimmer dB: 1.6897159297527053<br>';
window.onload = function() {
            document.getElementById('patientNameTitle').innerHTML = patientName + ' Report';
            document.getElementById('reportTitle').innerHTML = patientName + ' Report';
            document.getElementById('heartStats').innerHTML = heartStats;
            document.getElementById('breathStats').innerHTML = breathStats;
            document.getElementById('speechStats').innerHTML = speechStats;
        }