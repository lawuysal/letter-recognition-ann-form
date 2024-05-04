namespace letter_recognition_ann_form
{

    public partial class Form1 : Form
    {
        public NeuralNet network;

        public Form1()
        {
            InitializeComponent();

            // Initialize the network
            network = new NeuralNet(inputNodes: 35, hiddenNodes: 3, outputNodes: 5);


        }

        private void panel27_MouseClick(object sender, MouseEventArgs e)
        {
            Panel box = (Panel)sender;

            if (box.BackColor == Color.SteelBlue)
            {
                box.BackColor = Color.Lavender;
            }
            else
            {
                box.BackColor = Color.SteelBlue;
            }
        }

        private void panel37_Paint(object sender, PaintEventArgs e)
        {

        }

        private void start_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < 35; i++)
            {
                Panel panel = (Panel)drawPanel.Controls.Find("panel" + (i + 1), true).FirstOrDefault();
                NeuralNet.panels.Add(panel);
            }

            int panelIndex = 0;
            for (int i = 0; i < 7; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    if (NeuralNet.panels[panelIndex].BackColor == Color.SteelBlue)
                    {
                        NeuralNet.testInput[i, j] = 1;
                    }
                    else if (NeuralNet.panels[panelIndex].BackColor == Color.Lavender)
                    {
                        NeuralNet.testInput[i, j] = 0;
                    }
                    panelIndex++;
                }
            }

            double[] output = network.FeedForward(NeuralNet.testInput);

            outputA.Text = output[0].ToString();
            outputB.Text = output[1].ToString();
            outputC.Text = output[2].ToString();
            outputD.Text = output[3].ToString();
            outputE.Text = output[4].ToString();
        }

        private void button5_MouseClick(object sender, MouseEventArgs e)
        {
            double[,] samples = new double[5, 35]
            {
                {0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1},
                {1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0},
                {0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1},
                {1,1,1,0,0,1,0,0,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1,0,1,1,1,0,0},
                {1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1}
            };
            Button btn = (Button)sender;
            switch (btn.Text)
            {
                case "A":
                    for (int i = 0; i < 35; i++)
                    {
                        Panel panel = (Panel)drawPanel.Controls.Find("panel" + (i + 1), true).FirstOrDefault();
                        if (samples[0, i] == 1)
                        {
                            panel.BackColor = Color.SteelBlue;
                        }
                        else
                        {
                            panel.BackColor = Color.Lavender;
                        }
                    }
                    break;

                case "B":
                    for (int i = 0; i < 35; i++)
                    {
                        Panel panel = (Panel)drawPanel.Controls.Find("panel" + (i + 1), true).FirstOrDefault();
                        if (samples[1, i] == 1)
                        {
                            panel.BackColor = Color.SteelBlue;
                        }
                        else
                        {
                            panel.BackColor = Color.Lavender;
                        }
                    }
                    break;

                case "C":
                    for (int i = 0; i < 35; i++)
                    {
                        Panel panel = (Panel)drawPanel.Controls.Find("panel" + (i + 1), true).FirstOrDefault();
                        if (samples[2, i] == 1)
                        {
                            panel.BackColor = Color.SteelBlue;
                        }
                        else
                        {
                            panel.BackColor = Color.Lavender;
                        }
                    }
                    break;

                case "D":
                    for (int i = 0; i < 35; i++)
                    {
                        Panel panel = (Panel)drawPanel.Controls.Find("panel" + (i + 1), true).FirstOrDefault();
                        if (samples[3, i] == 1)
                        {
                            panel.BackColor = Color.SteelBlue;
                        }
                        else
                        {
                            panel.BackColor = Color.Lavender;
                        }
                    }
                    break;

                case "E":
                    for (int i = 0; i < 35; i++)
                    {
                        Panel panel = (Panel)drawPanel.Controls.Find("panel" + (i + 1), true).FirstOrDefault();
                        if (samples[4, i] == 1)
                        {
                            panel.BackColor = Color.SteelBlue;
                        }
                        else
                        {
                            panel.BackColor = Color.Lavender;
                        }
                    }
                    break;

                case "Clear":
                    for (int i = 0; i < 35; i++)
                    {
                        Panel panel = (Panel)drawPanel.Controls.Find("panel" + (i + 1), true).FirstOrDefault();
                        panel.BackColor = Color.Lavender;
                    }
                    break;
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            // Training data
            double[][,] trainingDataInputs = new double[5][,] {

                // A
                new double[,] {
                    {0,0,1,0,0},
                    {0,1,0,1,0},
                    {1,0,0,0,1},
                    {1,0,0,0,1},
                    {1,1,1,1,1},
                    {1,0,0,0,1},
                    {1,0,0,0,1} },

                // B
                new double[,] {
                    {1,1,1,1,0},
                    {1,0,0,0,1},
                    {1,0,0,0,1},
                    {1,1,1,1,0},
                    {1,0,0,0,1},
                    {1,0,0,0,1},
                    {1,1,1,1,0} },


                // C
                new double[,] {
                    {0,0,1,1,1},
                    {0,1,0,0,0},
                    {1,0,0,0,0},
                    {1,0,0,0,0},
                    {1,0,0,0,0},
                    {0,1,0,0,0},
                    {0,0,1,1,1} },

                // D
                new double[,] {
                    {1,1,1,0,0},
                    {1,0,0,1,0},
                    {1,0,0,0,1},
                    {1,0,0,0,1},
                    {1,0,0,0,1},
                    {1,0,0,1,0},
                    {1,1,1,0,0} },

                // E
                new double[,] {
                    {1,1,1,1,1},
                    {1,0,0,0,0},
                    {1,0,0,0,0},
                    {1,1,1,1,1},
                    {1,0,0,0,0},
                    {1,0,0,0,0},
                    {1,1,1,1,1} }
            };

            double[][] trainingDataTargets = new double[][] {
                new double[] {1,0,0,0,0},
                new double[] {0,1,0,0,0},
                new double[] {0,0,1,0,0},
                new double[] {0,0,0,1,0},
                new double[] {0,0,0,0,1} };

            // Train the network
            int epochs = 10000;
            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < trainingDataInputs.Length; j++)
                {
                    network.BackPropagation(trainingDataInputs[j], trainingDataTargets[j]);
                }
            }

            MessageBox.Show("Eðitim tamamlandý.");
        }

        private void serializeBtn_Click(object sender, EventArgs e)
        {
            NeuralNet.SerializeWeights();
            MessageBox.Show("Aðýrlýklar dosyaya kaydedildi.");
        }

        private void deserializeBtn_Click(object sender, EventArgs e)
        {
            NeuralNet.DeserializeWeights();
            MessageBox.Show("Aðýrlýklar dosyadan yüklendi.");
        }
    }

}
