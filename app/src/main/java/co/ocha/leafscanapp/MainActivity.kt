package co.ocha.leafscanapp

import android.app.Activity
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        var btnLive = findViewById(R.id.Live_detection) as Button
        var btnStatic = findViewById(R.id.static_detection) as Button

        btnLive.setOnClickListener {
//            val intent = Intent(this,Liv)
        }

        btnStatic.setOnClickListener {
            Utils.openImagePicker(this)
        }
    }

    override fun onResume() {
        super.onResume()
        if(!Utils.allPermissionsGranted(this)){
            Utils.requestRuntimePermissions(this)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {

        if(requestCode == Utils.REQUEST_CODE_PHOTO_LIBRARY && resultCode == Activity.RESULT_OK && data != null){
            val intent = Intent(this, StaticDetectionActivity::class.java)
            intent.data = data.data
            startActivity(intent)
        }else{
            super.onActivityResult(requestCode, resultCode, data)
        }
    }
}