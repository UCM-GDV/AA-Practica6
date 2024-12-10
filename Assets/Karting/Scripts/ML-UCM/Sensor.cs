using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Sensor : MonoBehaviour
{
    [SerializeField]
    private float distance;
    [SerializeField]
    private Color debugColor;

    private MeshRenderer render;

    private void Start()
    {
        render = GetComponent<MeshRenderer>();
    }

    public float Distance
    {
        get
        {
            return distance;
        }
    }

    public Color DebugColor
    {
        get
        {
            return debugColor;
        }
    }

    private void Update()
    {
        render.material.color = debugColor;
    }
}
