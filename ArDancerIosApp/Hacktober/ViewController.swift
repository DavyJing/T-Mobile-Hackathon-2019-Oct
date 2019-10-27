//
//  ViewController.swift
//  Hacktober
//
//  Created by KUANG YAN on 10/26/19.
//  Copyright © 2019 KUANG YAN. All rights reserved.
//

import UIKit
import SceneKit
import ARKit
import AVFoundation
import Vision
import Speech


class ViewController: UIViewController, ARSCNViewDelegate {
    
    @IBOutlet var sceneView: ARSCNView!
    
    var audioPlayer = AVAudioPlayer()
    
    var animations = [String: CAAnimation]()
    var nonCharacter = true
    let danceType = ["JazzFixed","HipHopFixed","BellydancingFixed"]
    let songs = ["OneStep","FunkYouUp","MoonRiver"]
    var songURL = [URL]()
    var idle = true
    var currentDance = 0
    
    let audioEngine = AVAudioEngine()
    let speechRecognizer:SFSpeechRecognizer? = SFSpeechRecognizer()
    let request = SFSpeechAudioBufferRecognitionRequest()
    var recognitionTask:SFSpeechRecognitionTask?
    
    var currentFaceAnchor: ARFaceAnchor?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        sceneView.delegate = self
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        
        //Prepare the audio play
        for url in songs{
            if let URL = Bundle.main.url(forResource: url, withExtension: "mp3"){
                songURL.append(URL)
            } else{
                print("Music Lost!")
            }
        }

        do{
            audioPlayer = try AVAudioPlayer(contentsOf: songURL[0])
            audioPlayer.prepareToPlay()
            
            let audioSession = AVAudioSession.sharedInstance()
            do{
                try audioSession.setCategory(AVAudioSession.Category.playback)
            } catch{
                print("Audio Session error")
            }
            
            
        } catch{
            print("mp3 file doesn's exist")
        }
        
//        recognizeSpeech()
//        // Tracking Face
//        resetTracking()
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()
        
        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else{return}
        let result = sceneView.hitTest(touch.location(in: sceneView), types: ARHitTestResult.ResultType.featurePoint)
        guard let hitResult = result.last else {return}
        let hitTransform = SCNMatrix4(hitResult.worldTransform)
        let hitVector = SCNVector3Make(hitTransform.m41, hitTransform.m42, hitTransform.m43)
        
        if nonCharacter{
            loadIdleAnimation(position: hitVector)
            nonCharacter = false
        } else{
            swapAnimation()
        }
    }
    
//    func resetTracking(){
//        //Face tracking
//        guard ARFaceTrackingConfiguration.isSupported else { return }
//        let configuration = ARFaceTrackingConfiguration()
//        configuration.isLightEstimationEnabled = true
//        sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
//    }
    
    func loadIdleAnimation (position: SCNVector3) {
        // Load the character in the idle animation
        let idleScene = SCNScene(named: "art.scnassets/idleFixed.dae")!
        
        // This node will be parent of all the animation models
        let node = SCNNode()
        
        // Add all the child nodes to the parent node
        for child in idleScene.rootNode.childNodes {
            node.addChildNode(child)
        }
        
        // Set up some properties
        node.position = position
        node.scale = SCNVector3(0.005, 0.005, 0.005)
        
        // Load all the DAE animations
        for dance in danceType{
            loadAnimation(withKey: dance, sceneName: "art.scnassets/"+dance , animationIdentifier: dance+"-1")
        }
        
        // Add the node to the scene
        sceneView.scene.rootNode.addChildNode(node)
        
       
        
    }
    
    func loadAnimation(withKey: String, sceneName:String, animationIdentifier:String) {
        let sceneURL = Bundle.main.url(forResource: sceneName, withExtension: "dae")
        let sceneSource = SCNSceneSource(url: sceneURL!, options: nil)
        
        
        if let animationObject = sceneSource?.entryWithIdentifier(animationIdentifier, withClass: CAAnimation.self) {
            // The animation will only play once
            animationObject.repeatCount = .infinity // Dance should never stop!!!
            // To create smooth transitions between animations
            animationObject.fadeInDuration = CGFloat(1)
            animationObject.fadeOutDuration = CGFloat(0.5)
            
            // Store the animation for later use
            animations[withKey] = animationObject
        }
    }
    
    func swapAnimation(){
        if idle{
            playAnimation(key: danceType[currentDance])
            idle = !idle
            audioPlayer.play()
        } else{
            stopAnimation(key: danceType[currentDance])
            currentDance = Int((currentDance+1)%danceType.count)
            switchSong()
            playAnimation(key: danceType[currentDance])
        }
        
    }
    
    func switchSong(){
        audioPlayer.stop()
        do{
            audioPlayer = try AVAudioPlayer(contentsOf: songURL[currentDance])
            audioPlayer.prepareToPlay()
        } catch{
            print("mp3 file doesn's exist")
        }
        audioPlayer.play()
    }
    
    func stopSong(){
        audioPlayer.stop()
    }
    
    func pauseSong(){
        audioPlayer.pause()
    }
    func resumeSong(){
        audioPlayer.play()
    }
    
    func playAnimation(key: String) {
        // Add the animation to start playing it right away
        if let animation = animations[key]{
            sceneView.scene.rootNode.addAnimation(animation, forKey: key)
        }
        
    }
    
    func stopAnimation(key: String) {
        // Stop the animation with a smooth transition
        sceneView.scene.rootNode.removeAnimation(forKey: key, blendOutDuration: CGFloat(0.5))
    }
    
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user
        
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        // Inform the user that the session has been interrupted, for example, by presenting an overlay
        
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required
        
    }
    
}

//extension ViewController{
//    func recognizeSpeech(){
//        let node = audioEngine.inputNode
//        let recordingFormat = node.outputFormat(forBus:0)
//        node.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat){buffer, _ in
//            self.request.append(buffer)
//        }
//        audioEngine.prepare()
//        do {
//            try audioEngine.start()
//        } catch{
//            return print(error)
//        }
//        guard let myRecognizer = SFSpeechRecognizer() else {return}
//        if !myRecognizer.isAvailable {return}
//        recognitionTask = speechRecognizer?.recognitionTask(with: request, resultHandler: {result,error in
//            if let result = result{
//                let bestString = result.bestTranscription.formattedString
//
//                var lastString: String = ""
//                for segment in result.bestTranscription.segments{
//                    let indexTo = bestString.index(bestString.startIndex, offsetBy: segment.substringRange.location)
//                    lastString = bestString.substring(from: indexTo)
//                }
//                self.checkForDances(resultString: lastString)
//            } else if let error = error{
//                    print(error)
//        }
//
//    })
//
//    }
//
//    func checkForDances(resultString: String){
//        switch resultString{
//        case "next":
//            switchSong()
//            swapAnimation()
//        case "stop":
//            stopSong()
//            stopAnimation(key: danceType[currentDance])
//
//        default:
//            break
//        }
//    }
//}
