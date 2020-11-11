var patientName = '';
var heartStats = '';
var breathStats = '';
var speechStats = '';

window.onload = function(){
    document.getElementById('patientNameTitle').innerHTML = patientName + 'Report';
    document.getElementById('reportTitle').innerHTML = patientName + 'Report';
    document.getElementById('heartStats').innerText = heartStats;
    document.getElementById('breathStats').innerText = breathStats;
    document.getElementById('speechStats').innerText = speechStats;
}